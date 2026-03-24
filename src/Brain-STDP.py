"""
Brain Server
============
Спайковый мозг как TCP сервис. Подключается любая среда.

Запуск:
    python brain_server.py --inputs 4 --outputs 2
    python brain_server.py --inputs 8 --outputs 5 --port 9000 --seed 42

Протокол (JSON строки, \n разделитель):
    Клиент → {"type": "step",   "inputs": [0.1, -0.5, 0.03, 0.8]}
    Сервер → {"type": "action", "output": 1, "confidence": 0.72}
    Клиент → {"type": "reward", "value": 1.0}
    Клиент → {"type": "reset"}          # необязательно, между эпизодами
    Клиент → {"type": "save", "path": "brain.pkl"}
    Клиент → {"type": "load", "path": "brain.pkl"}
    Сервер → {"type": "ok"}             # ответ на reset/save/load
    Сервер → {"type": "error", "msg": "..."} # при ошибке

Совместимость: Python 3.8+, numpy. pygame-ce опционально (только для --ui).
"""

import argparse
import json
import pickle
import socket
import threading
import time
import sys
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════
#  ПАРАМЕТРЫ СЕТИ (константы которые не зависят от задачи)
# ═══════════════════════════════════════════════════════

DT             = 0.1     # шаг симуляции, мс
TAU_M          = 50.0    # постоянная утечки мембраны | Насколько быстро нейрон "забывает" входные токи, чем больше значение — тем дольше нейрон остаётся активным после спайка. Малые значения (20-50 мс) делают сеть более динамичной, большие (100-200 мс) — более устойчивой.
V_THRESH       = 1.0
V_RESET        = 0.0
REFRAC         = 5       # тиков рефрактерного периода
NOISE_AMP      = 0.05    # фоновый шум

INPUT_RATE_HI  = 0.35    # вероятность спайка активного входа за тик
INPUT_RATE_LO  = 0.02    # фоновая вероятность

# Каждый входной "слот" кодируется группой нейронов
NEURONS_PER_IN = 5       # нейронов на один входной сигнал

# Выходные нейроны — группа на один выходной класс
NEURONS_PER_OUT = 10

# Внутренние нейроны: 80% excitatory, 20% inhibitory
INTERNAL_MULT   = 8      # N_INTERNAL = N_INPUT_NEURONS * INTERNAL_MULT (мин 160)

# Обучение
LR         = 0.035
LR_STDP    = 0.003   # learning rate для internal STDP (медленнее чем RL)
TAU_PRE    = 20.0    # затухание pre-synaptic trace для STDP
D_REWARD   = 2.0         # масштаб положительного модулятора
P_PUNISH   = 1.5         # масштаб отрицательного модулятора
W_MAX      = 4.0
W_MIN_IO   = 0.0         # input→output только положительные
TAU_MOD    = 35.0        # затухание модуляторов

# Тик-цикл на один вызов step()
TICKS_PER_STEP = 120     # мозг "думает" 120 тиков перед ответом


# ═══════════════════════════════════════════════════════
#  МОЗГ
# ═══════════════════════════════════════════════════════

class Brain:
    """
    Спайковая LIF сеть с reward-modulated Hebbian обучением.
    Автоматически масштабируется под любой n_inputs / n_outputs.
    """

    def __init__(self, n_inputs: int, n_outputs: int, seed: int = 0):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.seed      = seed

        rng = np.random.default_rng(seed)

        # Размеры
        ni  = n_inputs  * NEURONS_PER_IN   # число input нейронов
        no  = n_outputs * NEURONS_PER_OUT  # число output нейронов
        nin = max(160, ni * INTERNAL_MULT)  # внутренних
        nin_exc = int(nin * 0.8)
        nin_inh = nin - nin_exc

        self.NI  = ni
        self.NO  = no
        self.NIN = nin
        self.NIN_EXC = nin_exc
        self.NIN_INH = nin_inh
        self.NT  = ni + nin + no           # всего нейронов

        self.OUT_START = ni + nin
        self.INH_START = ni + nin_exc
        self.INH_END   = ni + nin

        # Состояние нейронов
        self.V       = np.zeros(self.NT)
        self.refrac  = np.zeros(self.NT, dtype=int)
        self.trace   = np.zeros(self.NT)
        self.spikes  = np.zeros(self.NT, dtype=bool)
        self.spikes_f = np.zeros(self.NT, dtype=np.float32)  # float версия без astype

        # Поisson-вероятности входных нейронов
        self.input_rates = np.full(ni, INPUT_RATE_LO)

        # Накопленная активность входов за step()
        self.in_accum   = np.zeros(ni)

        # Счётчики спайков выходных нейронов
        self.out_counts = np.zeros(no)

        # Синапсы
        self.W    = self._init_weights(rng)
        self.conn = self.W != 0

        # Предвыделенный буфер для STDP outer product (NIN×NIN) — без аллокаций в горячем цикле
        self._dW_buf = np.zeros((self.NIN, self.NIN), dtype=np.float32)

        # Модуляторы
        self.dopamine = 0.0
        self.pain     = 0.0
        self._k_mod   = np.exp(-DT / TAU_MOD)
        self._k_pre   = np.exp(-DT / TAU_PRE)

        # Pre-synaptic trace для reward-modulated STDP
        # Отдельный от визуального trace — своя константа затухания
        self.pre_trace = np.zeros(self.NT)

        # Статистика
        self.total_steps   = 0
        self.total_rewards = 0.0
        self.accuracy_hist = deque(maxlen=100)  # 1=correct, 0=wrong (если известно)

    # ── Инициализация весов ─────────────────────────────

    def _init_weights(self, rng):
        NI, NIN, NO, NT = self.NI, self.NIN, self.NO, self.NT
        NIN_EXC = self.NIN_EXC
        NIN_INH = self.NIN_INH
        OUT_S   = self.OUT_START
        INH_S   = self.INH_START
        INH_E   = self.INH_END

        W = np.zeros((NT, NT))

        # Input → Internal (25%)
        m = rng.random((NIN, NI)) < 0.25
        W[NI:NI+NIN, :NI] = rng.uniform(0.10, 0.35, (NIN, NI)) * m

        # Internal Exc → Internal (5%, слабые рекуррентные)
        m = rng.random((NIN, NIN_EXC)) < 0.05
        np.fill_diagonal(m, False)
        W[NI:NI+NIN, NI:NI+NIN_EXC] = rng.uniform(0.01, 0.08, (NIN, NIN_EXC)) * m

        # Internal Inh → Internal (20%, тормозные)
        m = rng.random((NIN, NIN_INH)) < 0.20
        W[NI:NI+NIN, INH_S:INH_E] = -rng.uniform(0.10, 0.30, (NIN, NIN_INH)) * m

        # Input → Output (45%, главный путь обучения)
        m = rng.random((NO, NI)) < 0.45
        W[OUT_S:, :NI] = rng.uniform(0.05, 0.20, (NO, NI)) * m

        # Internal → Output (10%)
        m = rng.random((NO, NIN)) < 0.10
        W[OUT_S:, NI:NI+NIN] = rng.uniform(0.02, 0.10, (NO, NIN)) * m

        # Lateral inhibition между output группами (winner-take-all)
        lat = 0.30
        for i in range(n_outputs := self.n_outputs):
            for j in range(n_outputs):
                if i == j:
                    continue
                gi_s = OUT_S + i * NEURONS_PER_OUT
                gi_e = gi_s + NEURONS_PER_OUT
                gj_s = OUT_S + j * NEURONS_PER_OUT
                gj_e = gj_s + NEURONS_PER_OUT
                W[gj_s:gj_e, gi_s:gi_e] = -lat

        np.fill_diagonal(W, 0)
        return W

    # ── Один тик симуляции ──────────────────────────────

    def _tick(self):
        NI     = self.NI
        NIN    = self.NIN
        OUT_S  = self.OUT_START
        INH_S  = self.INH_START
        INH_E  = self.INH_END
        int_s  = slice(NI, NI + NIN)

        # Poisson-спайки входных нейронов
        ext_fired = (np.random.random(NI) < self.input_rates) & (self.refrac[:NI] == 0)

        # Синаптический ток — spikes хранится как float32 чтобы не конвертировать
        I = self.W @ self.spikes_f

        # LIF update + шум (noise на все NT — дешевле чем active.sum() + fancy index)
        noise  = np.random.randn(self.NT) * NOISE_AMP
        active = self.refrac == 0
        self.V[active] += DT * (-self.V[active] / TAU_M + I[active] + noise[active])
        self.refrac[~active] -= 1

        # Спайки
        fired           = (self.V >= V_THRESH) & active
        fired[:NI]      = ext_fired
        self.V[fired]   = V_RESET
        self.refrac[fired] = REFRAC

        # Обновляем float-версию спайков (без astype в горячем пути)
        self.spikes_f[:] = 0.0
        self.spikes_f[fired] = 1.0
        self.spikes = fired

        # Счётчики (прямое сложение float, без astype)
        self.out_counts += self.spikes_f[OUT_S:]
        self.in_accum   += self.spikes_f[:NI]

        # Spike trace (визуализация)
        self.trace       *= 0.92
        self.trace[fired] += 1.0

        # Pre-synaptic trace (STDP)
        self.pre_trace          *= self._k_pre
        self.pre_trace[fired]   += 1.0

        # Reward-modulated STDP — только internal×internal
        mod = self.dopamine - self.pain
        if abs(mod) > 0.05:
            f_int  = self.spikes_f[int_s]          # (NIN,) float32
            pt_int = self.pre_trace[int_s]          # (NIN,) float64
            # outer без аллокации — пишем прямо в буфер
            np.outer(f_int, pt_int, out=self._dW_buf)
            self._dW_buf *= LR_STDP * mod
            # in-place mask + add — без временных массивов
            conn_int = self.conn[int_s, int_s]      # view, не копия
            self._dW_buf *= conn_int
            self.W[int_s, int_s] += self._dW_buf
            # Inhibitory остаются отрицательными
            self.W[int_s, INH_S:INH_E] = -np.abs(self.W[int_s, INH_S:INH_E])
            np.clip(self.W[int_s, int_s], -W_MAX, W_MAX, out=self.W[int_s, int_s])

        # Затухание модуляторов
        self.dopamine *= self._k_mod
        self.pain     *= self._k_mod

    # ── Публичный API ────────────────────────────────────

    def step(self, inputs: list[float]) -> tuple[int, float]:
        """
        Подать входы (нормализованные 0..1 или -1..1), получить (action, confidence).
        inputs: список длиной n_inputs
        """
        assert len(inputs) == self.n_inputs, \
            f"Ожидается {self.n_inputs} входов, получено {len(inputs)}"

        # Rate coding: число → вероятность Poisson (векторизованно, без цикла)
        arr = np.clip((np.asarray(inputs, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)
        rates = INPUT_RATE_LO + arr * (INPUT_RATE_HI - INPUT_RATE_LO)
        for i in range(self.n_inputs):
            s = i * NEURONS_PER_IN
            self.input_rates[s:s+NEURONS_PER_IN] = rates[i]

        self.out_counts[:] = 0.0
        self.in_accum[:]   = 0.0

        for _ in range(TICKS_PER_STEP):
            self._tick()

        self.input_rates[:] = INPUT_RATE_LO

        action, confidence = self._read_output()
        self.total_steps  += 1
        return action, confidence

    def reward(self, value: float):
        """
        Передать награду за последнее действие.
        value > 0: хорошо (дофамин)
        value < 0: плохо (боль/кортизол)
        value = 0: нейтрально
        """
        self.total_rewards += value

        if value > 0:
            self._apply_reward(self._last_winner, correct_direction=+1, magnitude=value)
            self.dopamine += D_REWARD * min(value, 1.0)
        elif value < 0:
            self._apply_reward(self._last_winner, correct_direction=-1, magnitude=-value)
            self.pain += P_PUNISH * min(-value, 1.0)

    def reset(self):
        """Сбросить состояние нейронов между эпизодами (веса сохраняются)."""
        self.V[:]          = 0.0
        self.refrac[:]     = 0
        self.trace[:]      = 0.0
        self.pre_trace[:]  = 0.0
        self.spikes[:]     = False
        self.spikes_f[:]   = 0.0
        self.out_counts[:] = 0.0
        self.in_accum[:]   = 0.0

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'n_inputs':  self.n_inputs,
                'n_outputs': self.n_outputs,
                'seed':      self.seed,
                'W':         self.W,
                'stats': {
                    'total_steps':   self.total_steps,
                    'total_rewards': self.total_rewards,
                }
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Brain':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        brain = cls(data['n_inputs'], data['n_outputs'], data['seed'])
        brain.W    = data['W']
        brain.conn = brain.W != 0
        brain.total_steps   = data['stats']['total_steps']
        brain.total_rewards = data['stats']['total_rewards']
        return brain

    # ── Внутренние методы ────────────────────────────────

    def _read_output(self) -> tuple[int, float]:
        counts = self.out_counts.reshape(self.n_outputs, NEURONS_PER_OUT).sum(axis=1)
        total  = counts.sum()

        if total == 0:
            winner = int(np.random.randint(0, self.n_outputs))
            self._last_winner = winner
            return winner, 0.0

        winner     = int(np.argmax(counts))
        confidence = float(counts[winner] / total)
        self._last_winner = winner
        return winner, confidence

    def _apply_reward(self, winner: int, correct_direction: int, magnitude: float):
        """
        Reward-modulated Hebbian update.
        Обновляет ДВА пути:
          1. input → output   (быстрый прямой путь)
          2. internal → output (медленный, использует то что STDP построил внутри)
        """
        NI, NIN = self.NI, self.NIN
        OUT_S   = self.OUT_START

        # Активность входов за этот step (накоплена в _tick)
        in_act  = self.in_accum / max(self.in_accum.max(), 1.0)

        # Активность internal нейронов — берём из pre_trace (след свежей активности)
        int_act = self.pre_trace[NI:NI+NIN]
        int_act = int_act / max(int_act.max(), 1.0)

        w_s = OUT_S + winner * NEURONS_PER_OUT
        w_e = w_s + NEURONS_PER_OUT

        delta_in  = LR       * magnitude * in_act[None, :]   # быстрый путь
        delta_int = LR * 0.4 * magnitude * int_act[None, :]  # медленный путь

        if correct_direction > 0:
            self.W[w_s:w_e, :NI]      += delta_in
            self.W[w_s:w_e, NI:NI+NIN] += delta_int
            for other in range(self.n_outputs):
                if other == winner:
                    continue
                o_s = OUT_S + other * NEURONS_PER_OUT
                o_e = o_s + NEURONS_PER_OUT
                self.W[o_s:o_e, :NI]       -= delta_in  * 0.5
                self.W[o_s:o_e, NI:NI+NIN] -= delta_int * 0.5
        else:
            self.W[w_s:w_e, :NI]       -= delta_in
            self.W[w_s:w_e, NI:NI+NIN] -= delta_int
            for other in range(self.n_outputs):
                if other == winner:
                    continue
                o_s = OUT_S + other * NEURONS_PER_OUT
                o_e = o_s + NEURONS_PER_OUT
                self.W[o_s:o_e, :NI]       += delta_in  * 0.6
                self.W[o_s:o_e, NI:NI+NIN] += delta_int * 0.6

        # input→output только положительные
        np.clip(self.W[OUT_S:, :NI], W_MIN_IO, W_MAX,
                out=self.W[OUT_S:, :NI])
        # internal→output могут быть отрицательными (торможение)
        np.clip(self.W[OUT_S:, NI:NI+NIN], -W_MAX, W_MAX,
                out=self.W[OUT_S:, NI:NI+NIN])

    def info(self) -> dict:
        return {
            'n_inputs':      self.n_inputs,
            'n_outputs':     self.n_outputs,
            'n_neurons':     self.NT,
            'n_synapses':    int(np.count_nonzero(self.W)),
            'total_steps':   self.total_steps,
            'total_rewards': round(self.total_rewards, 2),
            'dopamine':      round(self.dopamine, 3),
            'pain':          round(self.pain, 3),
        }


# ═══════════════════════════════════════════════════════
#  TCP СЕРВЕР
# ═══════════════════════════════════════════════════════

class BrainServer:
    def __init__(self, brain: Brain, host: str = '127.0.0.1', port: int = 7777):
        self.brain  = brain
        self.host   = host
        self.port   = port
        self._lock  = threading.Lock()   # несколько клиентов — защищаем мозг
        self._running = True

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        print(f"[Brain] Слушаю {self.host}:{self.port}")
        print(f"[Brain] Входов: {self.brain.n_inputs}  Выходов: {self.brain.n_outputs}")
        print(f"[Brain] Нейронов: {self.brain.NT}  Синапсов: {np.count_nonzero(self.brain.W)}")
        print(f"[Brain] Протокол: JSON lines.  Ctrl+C для остановки.\n")

        try:
            while self._running:
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                print(f"[Brain] Подключился: {addr[0]}:{addr[1]}")
                t = threading.Thread(target=self._handle, args=(conn, addr), daemon=True)
                t.start()
        except KeyboardInterrupt:
            print("\n[Brain] Остановка.")
        finally:
            srv.close()

    def _handle(self, conn: socket.socket, addr):
        buf = b''
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line)
                        resp = self._dispatch(msg)
                    except json.JSONDecodeError as e:
                        resp = {'type': 'error', 'msg': f'JSON parse error: {e}'}
                    except Exception as e:
                        resp = {'type': 'error', 'msg': str(e)}

                    conn.sendall((json.dumps(resp) + '\n').encode())
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            print(f"[Brain] Отключился: {addr[0]}:{addr[1]}")
            conn.close()

    def _dispatch(self, msg: dict) -> dict:
        t = msg.get('type')

        if t == 'step':
            inputs = msg['inputs']
            with self._lock:
                action, conf = self.brain.step(inputs)
            return {'type': 'action', 'output': action, 'confidence': round(conf, 4)}

        elif t == 'reward':
            value = float(msg['value'])
            with self._lock:
                self.brain.reward(value)
            return {'type': 'ok'}

        elif t == 'reset':
            with self._lock:
                self.brain.reset()
            return {'type': 'ok'}

        elif t == 'save':
            path = msg.get('path', 'brain_save.pkl')
            with self._lock:
                self.brain.save(path)
            print(f"[Brain] Сохранено: {path}")
            return {'type': 'ok', 'path': path}

        elif t == 'load':
            path = msg['path']
            with self._lock:
                loaded = Brain.load(path)
                # Проверяем совместимость
                if loaded.n_inputs != self.brain.n_inputs or \
                   loaded.n_outputs != self.brain.n_outputs:
                    return {'type': 'error',
                            'msg': f'Несовместимая конфигурация: '
                                   f'{loaded.n_inputs}→{loaded.n_outputs} '
                                   f'vs {self.brain.n_inputs}→{self.brain.n_outputs}'}
                self.brain.W    = loaded.W
                self.brain.conn = loaded.conn
                self.brain.total_steps   = loaded.total_steps
                self.brain.total_rewards = loaded.total_rewards
            print(f"[Brain] Загружено: {path}")
            return {'type': 'ok', 'path': path}

        elif t == 'info':
            with self._lock:
                return {'type': 'info', **self.brain.info()}

        else:
            return {'type': 'error', 'msg': f'Неизвестный тип: {t!r}'}


# ═══════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Brain Server — спайковый мозг как TCP сервис')
    p.add_argument('--inputs',  type=int, default=4,       help='Количество входных сигналов')
    p.add_argument('--outputs', type=int, default=2,       help='Количество выходных классов')
    p.add_argument('--host',    type=str, default='127.0.0.1')
    p.add_argument('--port',    type=int, default=7777)
    p.add_argument('--seed',    type=int, default=0)
    p.add_argument('--load',    type=str, default=None,    help='Загрузить сохранённый мозг')
    args = p.parse_args()

    if args.load:
        brain = Brain.load(args.load)
        print(f"[Brain] Загружен из {args.load}")
    else:
        brain = Brain(n_inputs=args.inputs, n_outputs=args.outputs, seed=args.seed)

    server = BrainServer(brain, host=args.host, port=args.port)
    server.start()