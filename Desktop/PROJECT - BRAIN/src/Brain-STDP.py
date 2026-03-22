"""
Brain Prototype v0.2  (pygame-ce)
===================================
Спайковая LIF сеть + reward-modulated Hebbian обучение.
Мозг живёт всегда. Тренер — один из процессов рядом.

Требования:
  pip install numpy pygame-ce

Управление:
  SPACE  — пауза
  ↑ / ↓  — скорость (тиков/кадр)
  R      — сбросить мозг
"""

import sys as _sys
try:
    import pygame as _pg_check
    if not getattr(_pg_check, 'IS_CE', False):
        print("Нужен pygame-ce, а не pygame.")
        print("  pip uninstall pygame")
        print("  pip install pygame-ce")
        _sys.exit(1)
except ImportError:
    print("pygame-ce не установлен.  pip install pygame-ce")
    _sys.exit(1)

import numpy as np
import pygame
import sys
import random
from collections import deque

# ═══════════════════════════════════════════════
#  ПАРАМЕТРЫ СЕТИ
# ═══════════════════════════════════════════════

N_INPUT    = 20
N_INTERNAL = 160   # 128 excitatory + 32 inhibitory
N_OUTPUT   = 20    # 10 → класс X,  10 → класс Y
N_TOTAL    = N_INPUT + N_INTERNAL + N_OUTPUT  # 200

N_EXC      = 128
N_INH      = 32
INH_START  = N_INPUT + N_EXC
INH_END    = N_INPUT + N_INTERNAL

OUT_START  = N_INPUT + N_INTERNAL
OUT_X      = slice(OUT_START, OUT_START + 10)
OUT_Y      = slice(OUT_START + 10, OUT_START + 20)

# ═══════════════════════════════════════════════
#  ПАРАМЕТРЫ LIF
# ═══════════════════════════════════════════════

DT        = 0.5      # мс
TAU_M     = 15.0     # постоянная утечки мембраны
V_THRESH  = 1.0
V_RESET   = 0.0
REFRAC    = 5        # тиков рефрактерного периода
NOISE_AMP = 0.05     # фоновый ток (держит сеть живой)

# ═══════════════════════════════════════════════
#  ПАРАМЕТРЫ ВХОДНОГО КОДИРОВАНИЯ
# ═══════════════════════════════════════════════

INPUT_RATE_ACTIVE = 0.35  # вероятность спайка за тик для активного входа
INPUT_RATE_BG     = 0.02  # фоновая вероятность (нейрон "молчит")
NEURONS_PER_BIT   = N_INPUT // 4   # 5 нейронов на один бит паттерна

# ═══════════════════════════════════════════════
#  ПАРАМЕТРЫ ОБУЧЕНИЯ
# ═══════════════════════════════════════════════

TAU_TRACE  = 30.0   # затухание spike trace
TAU_ELIG   = 60.0   # затухание eligibility trace
TAU_MOD    = 35.0   # затухание дофамина/боли

LR         = 0.035  # learning rate
A_PLUS     = 1.0    # STDP потенциация
A_MINUS    = 0.15   # STDP депрессия — асимметрия даёт net potentiation
W_MAX      = 4.0
W_MIN_IO   = 0.0   # input→output только положительные
W_MIN_REST = -1.0

D_REWARD   = 2.0
P_PUNISH   = 1.5

# ═══════════════════════════════════════════════
#  ПАРАМЕТРЫ ТРЕНИРОВКИ
# ═══════════════════════════════════════════════

T_THINK    = 120   # тиков на "думать"
T_DELAY    = 8     # тиков до модулятора
T_REWARD   = 50    # тиков действия модулятора

ACCURACY_WINDOW = 30

DATASET = [
    (np.array([0, 1, 0, 1]), 0),  # → X
    (np.array([1, 0, 0, 1]), 1),  # → Y
    (np.array([1, 1, 1, 1]), 1),  # → Y
    (np.array([0, 1, 1, 1]), 0),  # → X
]


# ═══════════════════════════════════════════════
#  МОЗГ
# ═══════════════════════════════════════════════

class Brain:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)

        # ── Состояние нейронов ──────────────────
        self.V       = np.zeros(N_TOTAL)
        self.refrac  = np.zeros(N_TOTAL, dtype=int)
        self.trace   = np.zeros(N_TOTAL)          # spike trace (pre-synaptic)
        self.spikes  = np.zeros(N_TOTAL, dtype=bool)

        # ── Синапсы ─────────────────────────────
        self.W    = self._init_weights(rng)
        self.conn = self.W != 0

        # ── Счётчики спайков выходов (читает Trainer) ──
        self.out_counts = np.zeros(N_OUTPUT)

        # ── Накопленная активность входов за триал ──
        self.in_accum   = np.zeros(N_INPUT)

        # ── Модуляторы ──────────────────────────
        self.dopamine = 0.0
        self.pain     = 0.0

        # ── Внешние активации входов (Poisson-вероятности) ──
        self.input_rates = np.full(N_INPUT, INPUT_RATE_BG)

        # ── Предвычисленные константы ───────────
        self._k_mod   = np.exp(-DT / TAU_MOD)

    # ── Инициализация весов ──────────────────────

    def _init_weights(self, rng):
        W = np.zeros((N_TOTAL, N_TOTAL))

        # Input → Internal: умеренные веса, 25% разреженность
        mask = rng.random((N_INTERNAL, N_INPUT)) < 0.25
        W[N_INPUT:N_INPUT+N_INTERNAL, :N_INPUT] = (
            rng.uniform(0.10, 0.35, (N_INTERNAL, N_INPUT)) * mask
        )

        # Internal Exc → Internal: очень слабые рекуррентные (5%)
        mask = rng.random((N_INTERNAL, N_EXC)) < 0.05
        np.fill_diagonal(mask, False)
        W[N_INPUT:N_INPUT+N_INTERNAL, N_INPUT:N_INPUT+N_EXC] = (
            rng.uniform(0.01, 0.08, (N_INTERNAL, N_EXC)) * mask
        )

        # Internal Inh → Internal: тормозные (20%)
        mask = rng.random((N_INTERNAL, N_INH)) < 0.20
        W[N_INPUT:N_INPUT+N_INTERNAL, INH_START:INH_END] = (
            -rng.uniform(0.10, 0.30, (N_INTERNAL, N_INH)) * mask
        )

        # Input → Output: прямые связи (45%), это главный путь для обучения
        mask = rng.random((N_OUTPUT, N_INPUT)) < 0.45
        W[OUT_START:, :N_INPUT] = (
            rng.uniform(0.05, 0.20, (N_OUTPUT, N_INPUT)) * mask
        )

        # Internal → Output: слабые (10%)
        mask = rng.random((N_OUTPUT, N_INTERNAL)) < 0.10
        W[OUT_START:, N_INPUT:N_INPUT+N_INTERNAL] = (
            rng.uniform(0.02, 0.10, (N_OUTPUT, N_INTERNAL)) * mask
        )

        # Lateral inhibition: X гасит Y, Y гасит X (winner-take-all)
        lat = 0.30
        for xi in range(OUT_START, OUT_START + 10):
            for yi in range(OUT_START + 10, OUT_START + 20):
                W[yi, xi] = -lat
                W[xi, yi] = -lat

        np.fill_diagonal(W, 0)
        return W

    # ── Один тик симуляции ───────────────────────

    def step(self):
        # 1. Poisson-спайки входных нейронов
        poisson_in = np.random.random(N_INPUT) < self.input_rates
        # Форсируем только те что не в рефрактерном периоде
        ext_fired = poisson_in & (self.refrac[:N_INPUT] == 0)

        # 2. Синаптический ток
        I = self.W @ self.spikes.astype(np.float32)

        # 3. LIF update (внутренние + выходные нейроны)
        noise  = np.random.randn(N_TOTAL) * NOISE_AMP
        active = self.refrac == 0

        # Входные нейроны двигаем через Poisson-флаги, не через ток
        self.V[active] += DT * (-self.V[active] / TAU_M + I[active] + noise[active])

        self.refrac[~active] -= 1

        # 4. Спайки — internal + output через потенциал, input через Poisson
        fired                  = (self.V >= V_THRESH) & active
        fired[:N_INPUT]        = ext_fired  # входные управляются извне

        self.V[fired]          = V_RESET
        self.refrac[fired]     = REFRAC
        self.spikes            = fired

        # 5. Счётчик output-спайков
        self.out_counts       += fired[OUT_START:].astype(float)

        # 6. Spike traces (визуализация — не участвует в обучении)
        self.trace            *= 0.92
        self.trace[fired]     += 1.0

        # 7. Накапливаем активность входных нейронов (для per-trial обучения)
        #    Тренер читает brain.in_accum при конце фазы think
        self.in_accum += fired[:N_INPUT].astype(float)

        # 8. Модуляторы применяют обновления (вызывается из Trainer напрямую)
        # (step не делает weight update сам — это задача Trainer.apply_reward)

        # 9. Затухание модуляторов
        self.dopamine *= self._k_mod
        self.pain     *= self._k_mod

    # ── API для Trainer ──────────────────────────

    def set_input(self, pattern):
        """4-битный паттерн → Poisson-вероятности входных нейронов"""
        for i, bit in enumerate(pattern):
            s = i * NEURONS_PER_BIT
            rate = INPUT_RATE_ACTIVE if bit else INPUT_RATE_BG
            self.input_rates[s:s+NEURONS_PER_BIT] = rate

    def clear_input(self):
        self.input_rates[:] = INPUT_RATE_BG

    def reset_accum(self):
        self.in_accum[:] = 0.0
        self.out_counts[:] = 0.0

    def apply_reward(self, winner_class: int, correct: bool):
        """
        Per-trial reward-modulated Hebbian update.
        Симметричный апдейт: winner усиляется/ослабляется, loser — зеркально.
        """
        in_act = self.in_accum / max(self.in_accum.max(), 1.0)

        w_s = OUT_START + winner_class * 10
        w_e = w_s + 10
        l_s = OUT_START + (1 - winner_class) * 10
        l_e = l_s + 10

        if correct:
            delta = LR * D_REWARD * in_act[None, :]
            self.W[w_s:w_e, :N_INPUT] += delta          # усилить winner (правильный)
            self.W[l_s:l_e, :N_INPUT] -= delta * 0.6    # ослабить loser
        else:
            delta = LR * P_PUNISH * in_act[None, :]
            self.W[w_s:w_e, :N_INPUT] -= delta          # ослабить winner (неправильный)
            self.W[l_s:l_e, :N_INPUT] += delta * 0.8    # усилить loser (правильный но проигравший)

        # Input→output только положительные
        np.clip(self.W[OUT_START:, :N_INPUT], W_MIN_IO, W_MAX,
                out=self.W[OUT_START:, :N_INPUT])

    def read_output(self):
        """Возвращает (winner: 0=X / 1=Y, confidence 0..1). Сбрасывает out_counts."""
        x = self.out_counts[:10].sum()
        y = self.out_counts[10:].sum()
        self.out_counts[:] = 0.0

        if x == 0 and y == 0:
            return random.randint(0, 1), 0.0

        winner     = 1 if y > x else 0
        confidence = abs(x - y) / (x + y)
        return winner, confidence


# ═══════════════════════════════════════════════
#  ТРЕНЕР
# ═══════════════════════════════════════════════

class Trainer:
    """
    State machine рядом с мозгом.
    Не режим — просто процесс который изредка выдаёт стимулы.
    """
    def __init__(self, brain: Brain):
        self.brain     = brain
        self.phase     = 'think'
        self.phase_t   = T_THINK
        self.pattern   = None
        self.expected  = None
        self.answer    = None
        self.conf      = 0.0
        self.trial_num = 0
        self.results   = deque(maxlen=ACCURACY_WINDOW)
        self.accuracy  = 0.0
        self.dataset   = list(DATASET)
        self.ds_idx    = 0
        self._start_trial()

    def _next_sample(self):
        if self.ds_idx >= len(self.dataset):
            random.shuffle(self.dataset)
            self.ds_idx = 0
        pat, exp = self.dataset[self.ds_idx]
        self.ds_idx += 1
        return pat, exp

    def _start_trial(self):
        self.pattern, self.expected = self._next_sample()
        self.answer  = None
        self.conf    = 0.0
        self.brain.set_input(self.pattern)
        self.brain.reset_accum()
        self.phase   = 'think'
        self.phase_t = T_THINK

    def step(self):
        self.phase_t -= 1

        if self.phase == 'think' and self.phase_t <= 0:
            self.answer, self.conf = self.brain.read_output()
            self.brain.clear_input()
            self.phase   = 'delay'
            self.phase_t = T_DELAY

        elif self.phase == 'delay' and self.phase_t <= 0:
            correct = (self.answer == self.expected)
            self.results.append(int(correct))
            self.trial_num += 1
            self.accuracy   = sum(self.results) / len(self.results)

            # Применяем reward — сразу меняем веса
            if self.answer is not None:
                self.brain.apply_reward(self.answer, correct)

            # Модуляторы для визуализации
            if correct:
                self.brain.dopamine += D_REWARD
            else:
                self.brain.pain     += P_PUNISH

            self.phase   = 'reward'
            self.phase_t = T_REWARD

        elif self.phase == 'reward' and self.phase_t <= 0:
            self._start_trial()


# ═══════════════════════════════════════════════
#  ВИЗУАЛИЗАЦИЯ
# ═══════════════════════════════════════════════

SW, SH = 1280, 720

PAL = {
    'bg':     ( 8,  10,  14),
    'panel':  (14,  18,  24),
    'border': (30,  40,  55),
    'text':   (185, 195, 210),
    'dim':    ( 60,  72,  90),
    'green':  ( 55, 215, 115),
    'red':    (225,  65,  55),
    'yellow': (238, 195,  55),
    'blue':   ( 75, 155, 250),
    'cyan':   ( 55, 215, 215),
    'purple': (155,  75, 250),
    'spike':  (255, 240, 100),
}


class Visualizer:
    def __init__(self, screen, font, small):
        self.screen = screen
        self.font   = font
        self.small  = small
        self.pos    = self._layout()

    def _layout(self):
        pos = np.zeros((N_TOTAL, 2), dtype=int)
        mg  = 30

        # Input — левая полоска
        for i in range(N_INPUT):
            pos[i] = [65, mg + i * (SH - 2*mg) // N_INPUT]

        # Internal — сетка
        cols, rows = 8, N_INTERNAL // 8
        x0 = 155
        for i in range(N_INTERNAL):
            c = i % cols
            r = i // cols
            pos[N_INPUT + i] = [
                x0 + c * 80,
                mg + 10 + r * (SH - 2*mg) // rows,
            ]

        # Output — правая полоска (оставляем место под HUD)
        x_out = SW - 290
        for i in range(N_OUTPUT):
            pos[N_INPUT + N_INTERNAL + i] = [
                x_out,
                mg + i * (SH - 2*mg) // N_OUTPUT,
            ]

        return pos

    def draw_neurons(self, brain):
        scr = self.screen

        for i in range(N_TOTAL):
            x, y = self.pos[i]
            sp   = brain.spikes[i]
            tr   = brain.trace[i]

            if i < N_INPUT:
                active = brain.input_rates[i] > INPUT_RATE_BG * 2
                if sp:
                    c, r = PAL['spike'], 7
                elif active:
                    t = min(1.0, tr)
                    c = (int(20 + 55*t), int(100 + 100*t), int(220 + 30*t))
                    r = 5
                else:
                    c, r = (18, 30, 55), 4
                pygame.draw.circle(scr, c, (x, y), r)

            elif i < N_INPUT + N_INTERNAL:
                is_inh = (i >= INH_START)
                if sp:
                    c, r = PAL['spike'], 5
                elif tr > 0.05:
                    t = min(1.0, tr * 0.8)
                    if is_inh:
                        c = (int(180*t), int(25*t), int(25*t))
                    else:
                        c = (int(15*t), int(185*t), int(70*t))
                    r = 3
                else:
                    c = (30, 12, 12) if is_inh else (12, 28, 16)
                    r = 2
                pygame.draw.circle(scr, c, (x, y), r)

            else:
                oi   = i - OUT_START
                is_x = oi < 10
                if sp:
                    c, r = PAL['spike'], 8
                elif tr > 0.05:
                    t = min(1.0, tr)
                    c = (int(240*t), int(195*t), 0) if is_x else (0, int(215*t), int(215*t))
                    r = 7
                else:
                    c = (38, 32, 8) if is_x else (8, 32, 38)
                    r = 5
                pygame.draw.circle(scr, c, (x, y), r)

        # Разделитель X / Y в output
        sep_y = self.pos[OUT_START + 10][1] - 9
        x_out = self.pos[OUT_START][0]
        pygame.draw.line(scr, PAL['border'], (x_out - 20, sep_y), (x_out + 20, sep_y), 1)

        # Подписи
        def lbl(t, cx, col):
            s = self.small.render(t, True, col)
            scr.blit(s, (cx - s.get_width()//2, 5))

        lbl("INPUT",    65,      PAL['blue'])
        lbl("INTERNAL", 155+280, PAL['green'])
        lbl("X / Y",    x_out,   PAL['yellow'])

    def draw_hud(self, brain, trainer, acc_hist, speed, paused):
        scr = self.screen
        px  = SW - 275
        py  = 22

        # Панель
        pygame.draw.rect(scr, PAL['panel'],  (px-8, py-8, 268, SH-30), border_radius=6)
        pygame.draw.rect(scr, PAL['border'], (px-8, py-8, 268, SH-30), 1, border_radius=6)

        def T(s, x, y, col=PAL['text'], fnt=None):
            scr.blit((fnt or self.font).render(s, True, col), (x, y))

        # ── Текущий паттерн ──────────────────
        if trainer.pattern is not None:
            bits = ''.join(map(str, trainer.pattern))
            exp  = 'XY'[trainer.expected]
            ans  = ('XY'[trainer.answer]
                    if trainer.answer is not None else '?')
            cor  = trainer.answer == trainer.expected

            T(f"Pattern : {bits}", px, py)
            T(f"Expected: {exp}",  px, py + 20)

            ans_col = PAL['green'] if cor else PAL['red']
            conf_s  = f"  ({trainer.conf:.0%})" if trainer.answer is not None else ""
            T(f"Answer  : {ans}{conf_s}", px, py + 40, ans_col)

        # ── Фаза ─────────────────────────────
        phase_col = {
            'think':  PAL['blue'],
            'delay':  PAL['yellow'],
            'reward': PAL['purple'],
        }.get(trainer.phase, PAL['dim'])
        T(f"Phase: {trainer.phase.upper()}", px, py + 68, phase_col, self.small)

        # ── Модуляторы ────────────────────────
        BY = py + 92
        BW = 210

        def bar(label, val, max_v, col, y):
            pygame.draw.rect(scr, (18, 18, 22), (px, y, BW, 14), border_radius=3)
            w = int(min(val / max(max_v, 0.001), 1.0) * BW)
            if w > 0:
                pygame.draw.rect(scr, col, (px, y, w, 14), border_radius=3)
            T(label, px + BW + 6, y, col, self.small)

        bar("DOP", brain.dopamine, D_REWARD, PAL['green'],  BY)
        bar("PAI", brain.pain,     P_PUNISH, PAL['red'],    BY + 18)

        # ── Статистика ────────────────────────
        AY = BY + 48
        acc_col = (PAL['green']  if trainer.accuracy >= 0.75 else
                   PAL['yellow'] if trainer.accuracy >= 0.50 else
                   PAL['red'])

        T(f"Trial   : {trainer.trial_num}",         px, AY)
        T(f"Accuracy: {trainer.accuracy:.1%}",       px, AY + 20, acc_col)

        # ── График accuracy ───────────────────
        GX, GY = px, AY + 50
        GW, GH = 230, 95
        pygame.draw.rect(scr, (11, 14, 19), (GX, GY, GW, GH))
        pygame.draw.rect(scr, PAL['border'], (GX, GY, GW, GH), 1)

        # 75% линия
        y75 = GY + GH - int(0.75 * GH)
        pygame.draw.line(scr, (35, 70, 35), (GX, y75), (GX+GW, y75))
        T("75%", GX + GW - 28, y75 - 11, (35, 70, 35), self.small)

        if len(acc_hist) > 1:
            n   = len(acc_hist)
            pts = [(GX + int(i * GW / max(n-1, 1)),
                    GY + GH - int(a * GH))
                   for i, a in enumerate(acc_hist)]
            pygame.draw.lines(scr, PAL['cyan'], False, pts, 2)

        T("accuracy over trials", GX + 2, GY + GH - 12, PAL['dim'], self.small)

        # ── Веса input→output ─────────────────
        WY = GY + GH + 18
        T("Weights  in→X  in→Y", px, WY, PAL['dim'], self.small)
        wx = brain.W[OUT_START:OUT_START+10,  :N_INPUT].mean()
        wy = brain.W[OUT_START+10:OUT_START+20, :N_INPUT].mean()
        col_wx = PAL['yellow'] if wx > wy else PAL['dim']
        col_wy = PAL['cyan']   if wy > wx else PAL['dim']
        T(f"         {wx:+.3f}  {wy:+.3f}", px, WY + 14, PAL['text'], self.small)

        # ── Подсказки ─────────────────────────
        hints = ["SPACE pause", "↑↓  speed", "R   reset"]
        hy    = SH - 72
        for h in hints:
            T(h, px, hy, PAL['dim'], self.small)
            hy += 14

        # Скорость / пауза
        status = "PAUSED" if paused else f"{speed}x"
        T(status, 8, SH - 16, PAL['dim'], self.small)


# ═══════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════

def main():
    pygame.init()
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Brain Prototype v0.2")
    clock  = pygame.time.Clock()

    font  = pygame.font.SysFont('monospace', 13)
    small = pygame.font.SysFont('monospace', 11)

    brain   = Brain(seed=42)
    trainer = Trainer(brain)
    vis     = Visualizer(screen, font, small)

    acc_hist = deque(maxlen=400)
    speed    = 8
    paused   = False

    while True:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            if ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_SPACE:
                    paused = not paused
                if ev.key == pygame.K_UP:
                    speed = min(speed + 4, 200)
                if ev.key == pygame.K_DOWN:
                    speed = max(speed - 4, 1)
                if ev.key == pygame.K_r:
                    brain   = Brain(seed=random.randint(0, 9999))
                    trainer = Trainer(brain)
                    acc_hist.clear()

        if not paused:
            for _ in range(speed):
                brain.step()
                trainer.step()
            if trainer.trial_num > 0:
                acc_hist.append(trainer.accuracy)

        screen.fill(PAL['bg'])
        vis.draw_neurons(brain)
        vis.draw_hud(brain, trainer, acc_hist, speed, paused)

        pygame.display.flip()
        clock.tick(60)


if __name__ == '__main__':
    main()