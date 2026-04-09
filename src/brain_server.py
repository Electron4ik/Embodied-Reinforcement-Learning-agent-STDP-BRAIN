"""
Brain Server v0.4
=================
Архитектурный рефактор. Не смена коэффициентов.

Что изменилось:
  - Прямой input→output ослаблен: выход опирается в основном на internal
  - Dual traces: короткий (шаг) + длинный (эпизод)
  - State-based critic по реальным признакам
  - Асимметричные апдейты с hard clip
  - Stochastic выбор с затуханием exploration
  - Чистая архитектура без мёртвых полей

Запуск:
    python brain_server.py --inputs 6 --outputs 2

Протокол: JSON lines, \\n разделитель.
"""

import argparse, json, pickle, socket, threading
import numpy as np

# ═══════════════════════════════════════════════════════
#  ПАРАМЕТРЫ СЕТИ
# ═══════════════════════════════════════════════════════

DT              = 0.5
TAU_M           = 20.0   # чуть длиннее — нейроны помнят вход дольше
V_THRESH        = 1.0
V_RESET         = 0.0
REFRAC          = 5
NOISE_AMP       = 0.008 

INPUT_RATE_HI   = 0.25
INPUT_RATE_LO   = 0.003

NEURONS_PER_IN  = 5
NEURONS_PER_OUT = 10
INTERNAL_MULT   = 6      # разумный баланс: не слишком мало, не огромный резервуар

# ── Обучение ─────────────────────────────────────────
LR              = 0.015  # медленно и стабильно
LR_STDP         = 0.005
LR_CRITIC       = 0.008
TAU_PRE         = 25.0

# Dual traces
TAU_SHORT       = 40.0   # короткий след — привязка к ближайшему reward
TAU_LONG        = 200.0  # длинный след — устойчивое закрепление стратегии

TAU_MOD         = 40.0
D_REWARD        = 1.5
P_PUNISH        = 0.5    # боль значительно мягче награды
W_MAX           = 3.0
W_MIN_IO        = 0.0

TICKS_PER_STEP  = 25     # чуть больше — лучше интегрирует сигнал

# ── Exploration decay ────────────────────────────────
EXPLORE_START   = 0.8    # начальная температура softmax
EXPLORE_MIN     = 0.1    # минимальная — почти argmax
EXPLORE_DECAY   = 0.995 # медленно убывает с опытом


# ═══════════════════════════════════════════════════════
#  МОЗГ
# ═══════════════════════════════════════════════════════

class Brain:
    def __init__(self, n_inputs: int, n_outputs: int, seed: int = 0):
        self.n_inputs  = n_inputs
        self.n_outputs = n_outputs
        self.seed      = seed

        rng = np.random.default_rng(seed)

        ni      = n_inputs  * NEURONS_PER_IN
        no      = n_outputs * NEURONS_PER_OUT
        nin     = max(120, ni * INTERNAL_MULT)
        nin_exc = int(nin * 0.8)
        nin_inh = nin - nin_exc

        self.NI      = ni
        self.NO      = no
        self.NIN     = nin
        self.NIN_EXC = nin_exc
        self.NIN_INH = nin_inh
        self.NT      = ni + nin + no

        self.OUT_START = ni + nin
        self.INH_START = ni + nin_exc
        self.INH_END   = ni + nin

        # ── Состояние ────────────────────────────────
        self.V        = np.zeros(self.NT)
        self.refrac   = np.zeros(self.NT, dtype=int)
        self.trace    = np.zeros(self.NT)       # визуализация
        self.spikes   = np.zeros(self.NT, dtype=bool)
        self.spikes_f = np.zeros(self.NT, dtype=np.float32)

        # ── Синапсы ──────────────────────────────────
        self.W    = self._init_weights(rng)
        self.conn = self.W != 0
        self._dW_buf = np.zeros((nin, nin), dtype=np.float32)

        # ── Входы / выходы ───────────────────────────
        self.input_rates = np.full(ni, INPUT_RATE_LO)
        self.in_accum    = np.zeros(ni)
        self.out_counts  = np.zeros(no)

        # ── Pre-trace (STDP) ─────────────────────────
        self.pre_trace = np.zeros(self.NT)
        self._k_pre    = np.exp(-DT / TAU_PRE)

        # ── Dual eligibility traces ──────────────────
        # short: быстрое затухание, привязка к ближайшему reward
        # long:  медленное, закрепляет стратегию между эпизодами
        feat_size    = ni + nin
        self.E_short = np.zeros((n_outputs, feat_size))
        self.E_long  = np.zeros((n_outputs, feat_size))
        self._k_short = np.exp(-DT * TICKS_PER_STEP / TAU_SHORT)
        self._k_long  = np.exp(-DT * TICKS_PER_STEP / TAU_LONG)

        # ── Memory (leaky integrator) ─────────────────
        self.memory = np.zeros(ni)

        # ── State critic ─────────────────────────────
        self.critic_w  = np.zeros(feat_size)
        self.last_feat = np.zeros(feat_size)
        self.last_v    = 0.0

        # ── Exploration ──────────────────────────────
        self.explore_temp = EXPLORE_START

        # ── Модуляторы ───────────────────────────────
        self.dopamine = 0.0
        self.pain     = 0.0
        self._k_mod   = np.exp(-DT / TAU_MOD)

        # ── Стат ─────────────────────────────────────
        self.total_steps   = 0
        self.total_rewards = 0.0
        self._last_winner  = 0
        self._last_conf    = 0.0

    # ── Веса ─────────────────────────────────────────

    def _init_weights(self, rng):
        NI, NIN, NO = self.NI, self.NIN, self.NO
        NT    = self.NT
        OUT_S = self.OUT_START
        INH_S = self.INH_START
        INH_E = self.INH_END
        W     = np.zeros((NT, NT))

        # Input → Internal (30%) — основной путь сигнала
        m = rng.random((NIN, NI)) < 0.30
        W[NI:NI+NIN, :NI] = rng.uniform(0.15, 0.45, (NIN, NI)) * m

        # Internal Exc → Internal (8%) — рекуррентная динамика
        m = rng.random((NIN, NIN)) < 0.08
        np.fill_diagonal(m, False)
        W[NI:NI+NIN, NI:NI+NIN] = rng.uniform(0.02, 0.10, (NIN, NIN)) * m

        # Internal Inh → Internal (25%)
        m = rng.random((NIN, self.NIN_INH)) < 0.25
        W[NI:NI+NIN, INH_S:INH_E] = -rng.uniform(0.12, 0.35, (NIN, self.NIN_INH)) * m

        # Internal → Output (20%) — главный путь к решению
        # Намеренно СИЛЬНЕЕ чем input→output
        m = rng.random((NO, NIN)) < 0.20
        W[OUT_S:, NI:NI+NIN] = rng.uniform(0.08, 0.22, (NO, NIN)) * m

        # Input → Output (15%) — ОСЛАБЛЕН: рефлекс не должен доминировать
        m = rng.random((NO, NI)) < 0.15
        W[OUT_S:, :NI] = rng.uniform(0.02, 0.08, (NO, NI)) * m

        # Lateral inhibition между output группами
        lat = 0.35
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                if i == j: continue
                gi_s = OUT_S + i * NEURONS_PER_OUT
                gj_s = OUT_S + j * NEURONS_PER_OUT
                W[gj_s:gj_s+NEURONS_PER_OUT, gi_s:gi_s+NEURONS_PER_OUT] = -lat

        np.fill_diagonal(W, 0)
        return W

    # ── Тик ──────────────────────────────────────────

    def _tick(self):
        NI    = self.NI
        NIN   = self.NIN
        OUT_S = self.OUT_START
        INH_S = self.INH_START
        INH_E = self.INH_END
        int_s = slice(NI, NI + NIN)

        ext_fired = (np.random.random(NI) < self.input_rates) & (self.refrac[:NI] == 0)

        I      = self.W @ self.spikes_f
        noise  = np.random.randn(self.NT) * NOISE_AMP
        active = self.refrac == 0
        self.V[active] += DT * (-self.V[active] / TAU_M + I[active] + noise[active])
        self.refrac[~active] -= 1

        fired            = (self.V >= V_THRESH) & active
        fired[:NI]       = ext_fired
        self.V[fired]    = V_RESET
        self.refrac[fired] = REFRAC

        self.spikes_f[:] = 0.0
        self.spikes_f[fired] = 1.0
        self.spikes = fired

        self.out_counts += self.spikes_f[OUT_S:]
        self.in_accum   += self.spikes_f[:NI]

        self.trace       *= 0.92
        self.trace[fired] += 1.0

        self.pre_trace        *= self._k_pre
        self.pre_trace[fired] += 1.0

        # STDP internal×internal — только при активном модуляторе
        mod = (self.dopamine - self.pain)
        if abs(mod) > 0.05:
            np.outer(self.spikes_f[int_s], self.pre_trace[int_s], out=self._dW_buf)
            self._dW_buf *= LR_STDP * mod
            self._dW_buf *= self.conn[int_s, int_s]
            self.W[int_s, int_s] += self._dW_buf
            self.W[int_s, INH_S:INH_E] = -np.abs(self.W[int_s, INH_S:INH_E])
            np.clip(self.W[int_s, int_s], -W_MAX, W_MAX, out=self.W[int_s, int_s])

        self.dopamine *= self._k_mod
        self.pain     *= self._k_mod

    # ── Публичный API ────────────────────────────────

    def step(self, inputs: list) -> tuple:
        assert len(inputs) == self.n_inputs, \
            f"Ожидается {self.n_inputs} входов, получено {len(inputs)}"

        arr   = np.clip((np.asarray(inputs, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)
        rates = INPUT_RATE_LO + arr * (INPUT_RATE_HI - INPUT_RATE_LO)
        for i in range(self.n_inputs):
            s = i * NEURONS_PER_IN
            self.input_rates[s:s+NEURONS_PER_IN] = rates[i]

        # Memory injection
        self.input_rates += self.memory * 0.12

        self.out_counts[:] = 0.0
        self.in_accum[:]   = 0.0

        for _ in range(TICKS_PER_STEP):
            self._tick()

        self.input_rates[:] = INPUT_RATE_LO

        # Обновляем память
        in_norm = self.in_accum / max(self.in_accum.max(), 1.0)
        self.memory = 0.88 * self.memory + 0.12 * in_norm

        action, confidence = self._read_output()

        # Dual eligibility traces
        self.E_short *= self._k_short
        self.E_long  *= self._k_long

        #in_act  = self.in_accum / max(self.in_accum.max(), 1.0)
        #int_act = self.pre_trace[self.NI:self.NI+self.NIN]
        #int_act = int_act / max(int_act.max(), 1.0)
        in_act  = self.in_accum / TICKS_PER_STEP
        int_act = self.pre_trace[self.NI:self.NI+self.NIN]
        int_act = int_act / max(int_act.max(), 1.0) # Правильная нормализация аналогового трейса
        feat    = np.concatenate([in_act, int_act])

        # Уверенный выбор оставляет более сильный след
        weight = 0.4 + confidence * 0.6
        self.E_short[action] += feat * weight
        self.E_long[action]  += feat * weight * 0.3

        # Critic
        self.last_feat = feat.copy()
        self.last_v    = float(self.critic_w @ feat)

        self.total_steps  += 1
        self._last_conf    = confidence
        return action, confidence

    def reward(self, value: float):
        """
        TD error → асимметричный апдейт через dual traces.
        Позитивные: усиляем через оба трейса.
        Негативные: только через короткий и очень мягко.
        """
        self.total_rewards += value

        td_error = value - self.last_v
        # Hard clip: один мисс не должен разнести всю стратегию
        td_error = float(np.clip(td_error, -2.5, 3.0))

        if abs(td_error) > 0.01:
            self.critic_w += LR_CRITIC * td_error * self.last_feat

        if td_error > 0:
            self.dopamine += D_REWARD * min(td_error, 2.0)
        else:
            self.pain += P_PUNISH * min(-td_error, 2.0)

        self._apply_reward(td_error)

        # Exploration decay — постепенно переходим к exploitation
        self.explore_temp = max(EXPLORE_MIN,
                                self.explore_temp * EXPLORE_DECAY)

    def reset(self):
        """
        Сброс между эпизодами.
        Короткий трейс — жёсткий сброс (не переносим мусор).
        Длинный трейс — ослабляем, но не убиваем (стратегия живёт).
        """
        self.dopamine = 0.0
        self.pain = 0.0
        self.V[:]          = 0.0
        self.refrac[:]     = 0
        self.trace[:]      = 0.0
        self.pre_trace[:]  = 0.0
        self.spikes[:]     = False
        self.spikes_f[:]   = 0.0
        self.out_counts[:] = 0.0
        self.in_accum[:]   = 0.0
        self.memory[:]     = 0.0

        self.E_short[:] = 0.0      # короткий — полный сброс
        self.E_long  *= 0.60       # длинный — ослабляем, стратегия остаётся

    def save(self, path: str):
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'n_inputs':    self.n_inputs,
                'n_outputs':   self.n_outputs,
                'seed':        self.seed,
                'W':           self.W,
                'critic_w':    self.critic_w,
                'E_long':      self.E_long,
                'explore':     self.explore_temp,
                'stats':       {'total_steps':   self.total_steps,
                                'total_rewards': self.total_rewards}
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Brain':
        import pickle
        with open(path, 'rb') as f:
            d = pickle.load(f)
        b = cls(d['n_inputs'], d['n_outputs'], d['seed'])
        b.W            = d['W']
        b.conn         = b.W != 0
        b.critic_w     = d.get('critic_w',  b.critic_w)
        b.E_long       = d.get('E_long',    b.E_long)
        b.explore_temp = d.get('explore',   b.explore_temp)
        b.total_steps  = d['stats']['total_steps']
        b.total_rewards = d['stats']['total_rewards']
        return b

    # ── Внутренние ───────────────────────────────────

    def _read_output(self) -> tuple:
        counts = self.out_counts.reshape(self.n_outputs, NEURONS_PER_OUT).sum(axis=1)
        total  = counts.sum()

        if total == 0:
            w = int(np.random.randint(0, self.n_outputs))
            self._last_winner = w
            return w, 0.0

        # Softmax с затухающей температурой
        temp   = self.explore_temp
        logits = counts / (temp + 1e-8)
        logits -= logits.max()
        probs  = np.exp(logits)
        probs /= probs.sum()

        w  = int(np.random.choice(self.n_outputs, p=probs))
        cf = float(counts[w] / total)
        self._last_winner = w
        return w, cf

    def _apply_reward(self, td_error: float):
        OUT_S  = self.OUT_START
        NI     = self.NI
        NIN    = self.NIN
        winner = self._last_winner

        magnitude = abs(td_error)
        # Confidence freeze: не ломаем то что работает
        if td_error > 0 and self._last_conf > 0.75:
            magnitude *= 0.25
        if magnitude < 1e-4:
            return

        w_s = OUT_S + winner * NEURONS_PER_OUT
        w_e = w_s + NEURONS_PER_OUT

        if td_error > 0:
            # 1. Сначала ОДИН РАЗ обновляем победителя (усиливаем то, что сработало)
            elig = self.E_short[winner] + 0.5 * self.E_long[winner]
            elig = np.clip(elig, 0, None)
            
            # Уверенный апдейт победителя
            delta_winner = LR * magnitude * elig[None, :]
            self.W[w_s:w_e, :NI+NIN] += delta_winner * self.conn[w_s:w_e, :NI+NIN]

            # 2. Теперь проходим по конкурентам и ОДИН РАЗ их ослабляем
            for other in range(self.n_outputs):
                if other == winner: continue
                
                o_s = OUT_S + other * NEURONS_PER_OUT
                o_e = o_s + NEURONS_PER_OUT
                
                elig_o = self.E_short[other]
                # Ослабляем конкурентов, чтобы выбор winner стал более явным
                self.W[o_s:o_e, :NI+NIN] -= LR * magnitude * 0.3 * elig_o[None, :] * self.conn[o_s:o_e, :NI+NIN]

        else:
            # 3. Наказание (td_error <= 0)
            # Берем меньше long_trace, чтобы один промах не убивал всю стратегию
            elig = self.E_short[winner] + 0.1 * self.E_long[winner] 
            delta_penalty = LR * magnitude * 0.8 * elig[None, :]
            
            # ВАЖНО: здесь тоже нужна маска conn!
            self.W[w_s:w_e, :NI+NIN] -= delta_penalty * self.conn[w_s:w_e, :NI+NIN]

        # Клипинг
        #np.clip(self.W[OUT_S:, :NI],       W_MIN_IO, W_MAX, out=self.W[OUT_S:, :NI])
        #np.clip(self.W[OUT_S:, NI:NI+NIN], -W_MAX,   W_MAX, out=self.W[OUT_S:, NI:NI+NIN])
        np.clip(self.W[OUT_S:, :NI], 0.0, W_MAX, out=self.W[OUT_S:, :NI])
        np.clip(self.W[OUT_S:, NI:NI+NIN], -W_MAX, W_MAX, out=self.W[OUT_S:, NI:NI+NIN])

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
            'critic_v':      round(self.last_v, 3),
            'explore':       round(self.explore_temp, 4),
        }


# ═══════════════════════════════════════════════════════
#  TCP СЕРВЕР
# ═══════════════════════════════════════════════════════

class BrainServer:
    def __init__(self, brain: Brain, host='127.0.0.1', port=7777):
        self.brain    = brain
        self.host     = host
        self.port     = port
        self._lock    = threading.Lock()
        self._running = True

    def start(self):
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(8)
        b = self.brain
        print(f"[Brain v0.4] {self.host}:{self.port}")
        print(f"  inputs={b.n_inputs} outputs={b.n_outputs}")
        print(f"  neurons={b.NT}  synapses={np.count_nonzero(b.W)}")
        print(f"  explore={b.explore_temp:.3f}  LR={LR}  LR_critic={LR_CRITIC}")
        print(f"  dual traces: short={TAU_SHORT}ms long={TAU_LONG}ms\n")
        try:
            while self._running:
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                print(f"  + {addr[0]}:{addr[1]}")
                threading.Thread(target=self._handle, args=(conn, addr), daemon=True).start()
        except KeyboardInterrupt:
            print("\n[Brain] Стоп.")
        finally:
            srv.close()

    def _handle(self, conn, addr):
        buf = b''
        try:
            while True:
                chunk = conn.recv(4096)
                if not chunk: break
                buf += chunk
                while b'\n' in buf:
                    line, buf = buf.split(b'\n', 1)
                    line = line.strip()
                    if not line: continue
                    try:
                        resp = self._dispatch(json.loads(line))
                    except json.JSONDecodeError as e:
                        resp = {'type': 'error', 'msg': f'JSON: {e}'}
                    except Exception as e:
                        resp = {'type': 'error', 'msg': str(e)}
                    conn.sendall((json.dumps(resp) + '\n').encode())
        except (ConnectionResetError, BrokenPipeError):
            pass
        finally:
            print(f"  - {addr[0]}:{addr[1]}")
            conn.close()

    def _dispatch(self, msg: dict) -> dict:
        t = msg.get('type')
        if t == 'step':
            with self._lock:
                action, conf = self.brain.step(msg['inputs'])
            return {'type': 'action', 'output': action, 'confidence': round(conf, 4)}
        elif t == 'reward':
            with self._lock:
                self.brain.reward(float(msg['value']))
            return {'type': 'ok'}
        elif t == 'reset':
            with self._lock:
                self.brain.reset()
            return {'type': 'ok'}
        elif t == 'save':
            path = msg.get('path', 'brain_save.pkl')
            with self._lock:
                self.brain.save(path)
            print(f"  saved: {path}")
            return {'type': 'ok', 'path': path}
        elif t == 'load':
            path = msg['path']
            with self._lock:
                loaded = Brain.load(path)
                if loaded.n_inputs != self.brain.n_inputs or \
                   loaded.n_outputs != self.brain.n_outputs:
                    return {'type': 'error', 'msg': 'Несовместимая конфигурация'}
                self.brain.W            = loaded.W
                self.brain.conn         = loaded.conn
                self.brain.critic_w     = loaded.critic_w
                self.brain.E_long       = loaded.E_long
                self.brain.explore_temp = loaded.explore_temp
                self.brain.total_steps  = loaded.total_steps
                self.brain.total_rewards = loaded.total_rewards
            print(f"  loaded: {path}")
            return {'type': 'ok', 'path': path}
        elif t == 'info':
            with self._lock:
                return {'type': 'info', **self.brain.info()}
        return {'type': 'error', 'msg': f'Неизвестный тип: {t!r}'}


# ═══════════════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--inputs',  type=int, default=6)
    p.add_argument('--outputs', type=int, default=2)
    p.add_argument('--host',    type=str, default='127.0.0.1')
    p.add_argument('--port',    type=int, default=7777)
    p.add_argument('--seed',    type=int, default=0)
    p.add_argument('--load',    type=str, default=None)
    args = p.parse_args()

    if args.load:
        brain = Brain.load(args.load)
        print(f"[Brain] Загружен: {args.load}")
    else:
        brain = Brain(n_inputs=args.inputs, n_outputs=args.outputs, seed=args.seed)

    BrainServer(brain, host=args.host, port=args.port).start()