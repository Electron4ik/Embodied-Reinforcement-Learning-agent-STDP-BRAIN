"""
Brain Server v0.5  —  Actor-Critic с настоящим TD
==================================================
Запуск:
    python brain_server.py --inputs 6 --outputs 3

Ключевые изменения vs v0.4:
  - Настоящий TD: r + γ·V(s') - V(s)  (не просто r - V(s))
  - Memory как отдельный признак в feat, не инжекция в input_rates
  - Асимметрия: pos update сильнее, neg — мягче
  - Exploration decay по шагам, не по reward()
  - Чистый save/load без мёртвых полей
"""

import argparse, json, pickle, socket, threading
import numpy as np

# ═══════════════════════════════════════════════
#  ПАРАМЕТРЫ
# ═══════════════════════════════════════════════

DT             = 0.5
TAU_M          = 20.0
V_THRESH       = 1.0
V_RESET        = 0.0
REFRAC         = 5
NOISE_AMP      = 0.008

INPUT_RATE_HI  = 0.25
INPUT_RATE_LO  = 0.003

NEURONS_PER_IN  = 5
NEURONS_PER_OUT = 10
INTERNAL_MULT   = 6

# Обучение
LR             = 0.012
LR_STDP        = 0.004
LR_CRITIC      = 0.01
TAU_PRE        = 25.0
GAMMA          = 0.97      # discount factor для TD

TAU_SHORT      = 40.0
TAU_LONG       = 250.0
TAU_MOD        = 40.0

D_REWARD       = 1.5
P_PUNISH       = 0.4       # боль мягче награды

W_MAX          = 3.0
W_MIN_IO       = 0.0

TICKS_PER_STEP = 25

EXPLORE_START  = 0.9
EXPLORE_MIN    = 0.12
EXPLORE_DECAY  = 0.99985    # по шагам, не по reward


# ═══════════════════════════════════════════════
#  МОЗГ
# ═══════════════════════════════════════════════

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

        self.NI  = ni;  self.NO  = no;  self.NIN = nin
        self.NIN_EXC = nin_exc;  self.NIN_INH = nin_inh
        self.NT  = ni + nin + no
        self.OUT_START = ni + nin
        self.INH_START = ni + nin_exc
        self.INH_END   = ni + nin

        # Нейроны
        self.V        = np.zeros(self.NT)
        self.refrac   = np.zeros(self.NT, dtype=int)
        self.trace    = np.zeros(self.NT)
        self.spikes   = np.zeros(self.NT, dtype=bool)
        self.spikes_f = np.zeros(self.NT, dtype=np.float32)

        # Синапсы
        self.W        = self._init_weights(rng)
        self.conn     = self.W != 0
        self._dW_buf  = np.zeros((nin, nin), dtype=np.float32)

        self.value_w = np.zeros(self.n_inputs + 1, dtype=np.float32)  # critic по obs + bias
        self.prev_obs = np.zeros(self.n_inputs, dtype=np.float32)
        self.prev_value_feat = np.zeros(self.n_inputs + 1, dtype=np.float32)
        self.prev_v = 0.0

        # Входы
        self.input_rates = np.full(ni, INPUT_RATE_LO)
        self.in_accum    = np.zeros(ni)
        self.out_counts  = np.zeros(no)

        # Pre-trace (STDP)
        self.pre_trace = np.zeros(self.NT)
        self._k_pre    = np.exp(-DT / TAU_PRE)

        # Memory — leaky integrator internal активности (отдельно от входов!)
        # Размер = NIN, подаётся в feat вместе с in_act и int_act
        self.memory    = np.zeros(nin)

        # Feat размер: in_act(ni) + int_act(nin) + memory(nin)
        feat_size      = ni + nin + nin
        self.feat_size = feat_size

        # Dual eligibility traces
        self.E_short  = np.zeros((n_outputs, feat_size))
        self.E_long   = np.zeros((n_outputs, feat_size))
        self._k_short = np.exp(-DT * TICKS_PER_STEP / TAU_SHORT)
        self._k_long  = np.exp(-DT * TICKS_PER_STEP / TAU_LONG)

        # Critic: линейная аппроксимация V(s)
        self.critic_w   = np.zeros(feat_size)
        self.last_feat  = np.zeros(feat_size)
        self.last_v     = 0.0       # V(s_current) — нужен для TD
        self.next_v     = 0.0       # V(s_next)   — вычисляется в step()

        # Exploration
        self.explore_temp = EXPLORE_START

        # Модуляторы
        self.dopamine  = 0.0
        self.pain      = 0.0
        self._k_mod    = np.exp(-DT / TAU_MOD)

        # Статистика
        self.total_steps   = 0
        self.total_rewards = 0.0
        self._last_winner  = 0
        self._last_conf    = 0.0
        self._episode_done = False  # флаг для terminal TD

    # ── Веса ────────────────────────────────────

    def _init_weights(self, rng):
        NI, NIN, NO = self.NI, self.NIN, self.NO
        OUT_S = self.OUT_START
        INH_S = self.INH_START
        INH_E = self.INH_END
        W     = np.zeros((self.NT, self.NT))

        # Input → Internal (30%) — основной путь
        m = rng.random((NIN, NI)) < 0.30
        W[NI:NI+NIN, :NI] = rng.uniform(0.15, 0.45, (NIN, NI)) * m

        # Internal Exc → Internal (8%)
        m = rng.random((NIN, NIN)) < 0.08
        np.fill_diagonal(m, False)
        W[NI:NI+NIN, NI:NI+NIN] = rng.uniform(0.02, 0.10, (NIN, NIN)) * m

        # Internal Inh → Internal (25%)
        m = rng.random((NIN, self.NIN_INH)) < 0.25
        W[NI:NI+NIN, INH_S:INH_E] = -rng.uniform(0.12, 0.35, (NIN, self.NIN_INH)) * m

        # Internal → Output (20%) — СИЛЬНЕЕ чем input→output
        m = rng.random((NO, NIN)) < 0.20
        W[OUT_S:, NI:NI+NIN] = rng.uniform(0.08, 0.22, (NO, NIN)) * m

        # Input → Output (15%) — ослаблен, рефлекс не доминирует
        m = rng.random((NO, NI)) < 0.15
        W[OUT_S:, :NI] = rng.uniform(0.02, 0.08, (NO, NI)) * m

        # Lateral inhibition
        lat = 0.35
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                if i == j: continue
                gi = OUT_S + i * NEURONS_PER_OUT
                gj = OUT_S + j * NEURONS_PER_OUT
                W[gj:gj+NEURONS_PER_OUT, gi:gi+NEURONS_PER_OUT] = -lat

        np.fill_diagonal(W, 0)
        return W

    # ── Тик ─────────────────────────────────────

    def _tick(self):
        NI, NIN = self.NI, self.NIN
        OUT_S   = self.OUT_START
        INH_S   = self.INH_START
        INH_E   = self.INH_END
        int_s   = slice(NI, NI + NIN)

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

        self.trace        *= 0.92
        self.trace[fired] += 1.0

        self.pre_trace        *= self._k_pre
        self.pre_trace[fired] += 1.0

        mod = self.dopamine - self.pain
        if abs(mod) > 0.05:
            np.outer(self.spikes_f[int_s], self.pre_trace[int_s], out=self._dW_buf)
            self._dW_buf *= LR_STDP * mod * self.conn[int_s, int_s]
            self.W[int_s, int_s] += self._dW_buf
            self.W[int_s, INH_S:INH_E] = -np.abs(self.W[int_s, INH_S:INH_E])
            np.clip(self.W[int_s, int_s], -W_MAX, W_MAX, out=self.W[int_s, int_s])

        self.dopamine *= self._k_mod
        self.pain     *= self._k_mod

    # ── Публичный API ───────────────────────────

    def _value_feat(self, obs):
        x = np.asarray(obs, dtype=np.float32)
        x = np.clip((x + 1.0) / 2.0, 0.0, 1.0)
        return np.concatenate([x, np.array([1.0], dtype=np.float32)])

    def step(self, inputs: list) -> tuple:
        assert len(inputs) == self.n_inputs, \
            f"Ожидается {self.n_inputs} входов, получено {len(inputs)}"

        arr   = np.clip((np.asarray(inputs, dtype=np.float32) + 1.0) / 2.0, 0.0, 1.0)
        rates = INPUT_RATE_LO + arr * (INPUT_RATE_HI - INPUT_RATE_LO)
        for i in range(self.n_inputs):
            s = i * NEURONS_PER_IN
            self.input_rates[s:s+NEURONS_PER_IN] = rates[i]

        self.out_counts[:] = 0.0
        self.in_accum[:]   = 0.0

        for _ in range(TICKS_PER_STEP):
            self._tick()

        self.input_rates[:] = INPUT_RATE_LO

        # Нормировка активности
        in_act  = self.in_accum / TICKS_PER_STEP
        int_act = self.pre_trace[self.NI:self.NI+self.NIN].copy()
        mx = int_act.max()
        if mx > 0: int_act /= mx

        self.prev_obs[:] = arr
        self.prev_value_feat = self._value_feat(arr)
        self.prev_v = float(self.value_w @ self.prev_value_feat)

        # Memory: отдельный leaky integrator internal активности
        # НЕ инжектируется в input_rates — это признак в feat
        self.memory = 0.85 * self.memory + 0.15 * int_act
        mem_norm    = self.memory / max(self.memory.max(), 1.0)

        # Признаки состояния: [input_activity, internal_activity, memory]
        feat = np.concatenate([in_act, int_act, mem_norm])

        # Сохраняем V(s_current) перед выбором действия
        self.last_feat = feat.copy()
        self.last_v    = float(self.critic_w @ feat)

        action, confidence = self._read_output()

        # Dual eligibility traces
        self.E_short *= self._k_short
        self.E_long  *= self._k_long
        weight = 0.4 + confidence * 0.6
        self.E_short[action] += feat * weight
        self.E_long[action]  += feat * weight * 0.25

        # Exploration decay по шагам
        self.explore_temp = max(EXPLORE_MIN, self.explore_temp * EXPLORE_DECAY)

        # Exploration: с вероятностью explore_temp выбираем случайное действие, иначе — по активности выходных нейронов
        if np.random.random() < self.explore_temp:
            action = int(np.random.randint(0, self.n_outputs))
            confidence = 0.0
        else:
            counts = self.out_counts.reshape(self.n_outputs, NEURONS_PER_OUT).sum(axis=1)
            total = counts.sum()
            if total == 0:
                action = int(np.random.randint(0, self.n_outputs))
                confidence = 0.0
            else:
                temp = max(0.25, 0.35 + 0.65 * self.explore_temp)
                logits = counts / (temp + 1e-8)
                logits -= logits.max()
                probs = np.exp(logits)
                probs /= probs.sum()
                action = int(np.random.choice(self.n_outputs, p=probs))
                confidence = float(counts[action] / total)

        self.total_steps  += 1
        self._last_conf    = confidence
        self._episode_done = False
        return action, confidence

    def reward(self, value: float, terminal: bool = False, next_inputs=None):
        self.total_rewards += value

        if terminal:
            next_v = 0.0
        elif next_inputs is None:
            next_v = self.prev_v
        else:
            next_feat = self._value_feat(next_inputs)
            next_v = float(self.value_w @ next_feat)

        td = value + GAMMA * next_v - self.prev_v
        td = float(np.clip(td, -1.5, 3.0))

        # critic update
        if abs(td) > 0.01:
            self.value_w += LR_CRITIC * td * self.prev_value_feat
            norm = np.linalg.norm(self.value_w)
            if norm > 10.0:
                self.value_w *= 10.0 / norm

        # модуляторы
        if td > 0:
            self.dopamine += D_REWARD * min(td, 2.0)
        else:
            self.pain += P_PUNISH * min(-td, 2.0)

        self._apply_reward(td)

    def reset(self):
        """Короткий трейс — полный сброс. Длинный — ослабить."""
        self.V[:]          = 0.0
        self.refrac[:]     = 0
        self.trace[:]      = 0.0
        self.pre_trace[:]  = 0.0
        self.spikes[:]     = False
        self.spikes_f[:]   = 0.0
        self.out_counts[:] = 0.0
        self.in_accum[:]   = 0.0
        self.memory[:]     = 0.0
        self.dopamine      = 0.0
        self.pain          = 0.0
        self.E_short[:]    = 0.0
        self.E_long       *= 0.55
        self.prev_obs[:] = 0.0
        self.prev_value_feat[:] = 0.0
        self.prev_v = 0.0

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'version':   'v0.5',
                'n_inputs':  self.n_inputs,
                'n_outputs': self.n_outputs,
                'seed':      self.seed,
                'W':         self.W,
                'critic_w':  self.critic_w,
                'E_long':    self.E_long,
                'value_w':   self.value_w,
                'explore':   self.explore_temp,
                'stats':     {'total_steps':   self.total_steps,
                              'total_rewards': self.total_rewards}
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Brain':
        with open(path, 'rb') as f:
            d = pickle.load(f)
        b = cls(d['n_inputs'], d['n_outputs'], d['seed'])
        b.W             = d['W']
        b.conn          = b.W != 0
        b.critic_w      = d.get('critic_w',  b.critic_w)
        b.E_long        = d.get('E_long',    b.E_long)
        b.explore_temp  = d.get('explore',   b.explore_temp)
        b.total_steps   = d['stats']['total_steps']
        b.value_w       = d.get('value_w', b.value_w)
        b.total_rewards = d['stats']['total_rewards']
        return b

    # ── Внутренние ──────────────────────────────

    def _read_output(self) -> tuple:
        counts = self.out_counts.reshape(self.n_outputs, NEURONS_PER_OUT).sum(axis=1)
        total  = counts.sum()

        if total == 0:
            w = int(np.random.randint(0, self.n_outputs))
            self._last_winner = w
            return w, 0.0

        temp   = self.explore_temp
        logits = counts / (temp + 1e-8)
        logits -= logits.max()
        probs   = np.exp(logits)
        probs  /= probs.sum()

        w  = int(np.random.choice(self.n_outputs, p=probs))
        cf = float(counts[w] / total)
        self._last_winner = w
        return w, cf

    def _apply_reward(self, td: float):
        OUT_S  = self.OUT_START
        NI     = self.NI
        NIN    = self.NIN
        winner = self._last_winner
        
        # Граница физических нейронов (входы + внутренние)
        PHYS_N = NI + NIN 

        mag = abs(td)
        if td > 0 and self._last_conf > 0.80:
            mag *= 0.20
        if td < 0:
            mag *= 0.35
        if mag < 1e-4:
            return
        
        w_s = OUT_S + winner * NEURONS_PER_OUT
        w_e = w_s + NEURONS_PER_OUT

        if td > 0:
            # Берем только первые PHYS_N элементов из трейсов (обрезаем память)
            elig = np.clip(self.E_short[winner, :PHYS_N] + 0.4 * self.E_long[winner, :PHYS_N], 0, None)
            delta = LR * mag * elig[None, :]
            self.W[w_s:w_e, :PHYS_N] += delta * self.conn[w_s:w_e, :PHYS_N]
            
            for other in range(self.n_outputs):
                if other == winner: continue
                o_s = OUT_S + other * NEURONS_PER_OUT
                o_e = o_s + NEURONS_PER_OUT
                # Тут тоже обрезаем до PHYS_N
                elig_o = np.clip(self.E_short[other, :PHYS_N], 0, None)
                self.W[o_s:o_e, :PHYS_N] -= (LR * mag * 0.25 * elig_o[None, :] 
                                              * self.conn[o_s:o_e, :PHYS_N])
        else:
            # И тут обрезаем до PHYS_N
            elig = np.clip(self.E_short[winner, :PHYS_N], 0, None)
            delta = LR * mag * 0.5 * elig[None, :]
            self.W[w_s:w_e, :PHYS_N] -= delta * self.conn[w_s:w_e, :PHYS_N]

        # Клипинг
        np.clip(self.W[OUT_S:, :NI],       W_MIN_IO, W_MAX, out=self.W[OUT_S:, :NI])
        np.clip(self.W[OUT_S:, NI:PHYS_N], -W_MAX,   W_MAX, out=self.W[OUT_S:, NI:PHYS_N])

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


# ═══════════════════════════════════════════════
#  TCP СЕРВЕР
# ═══════════════════════════════════════════════

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
        print(f"[Brain v0.5]  {self.host}:{self.port}")
        print(f"  inputs={b.n_inputs}  outputs={b.n_outputs}")
        print(f"  neurons={b.NT}  synapses={np.count_nonzero(b.W)}")
        print(f"  feat_size={b.feat_size}  (in+int+mem)")
        print(f"  TD γ={GAMMA}  LR={LR}  LR_critic={LR_CRITIC}")
        print(f"  explore={b.explore_temp:.3f}→{EXPLORE_MIN}  decay={EXPLORE_DECAY}/step\n")
        try:
            while self._running:
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                print(f"  + {addr[0]}:{addr[1]}")
                threading.Thread(target=self._handle,
                                 args=(conn, addr), daemon=True).start()
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
            terminal = bool(msg.get('terminal', False))
            next_inputs = msg.get('next_inputs', None)
            with self._lock:
                self.brain.reward(float(msg['value']), terminal=terminal, next_inputs=next_inputs)
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
                ld = Brain.load(path)
                if ld.n_inputs != self.brain.n_inputs or \
                   ld.n_outputs != self.brain.n_outputs:
                    return {'type': 'error', 'msg': 'Несовместимая конфигурация'}
                self.brain.W            = ld.W
                self.brain.conn         = ld.conn
                self.brain.critic_w     = ld.critic_w
                self.brain.E_long       = ld.E_long
                self.brain.explore_temp = ld.explore_temp
                self.brain.total_steps  = ld.total_steps
                self.brain.total_rewards = ld.total_rewards
            print(f"  loaded: {path}")
            return {'type': 'ok', 'path': path}

        elif t == 'info':
            with self._lock:
                return {'type': 'info', **self.brain.info()}

        return {'type': 'error', 'msg': f'Неизвестный тип: {t!r}'}


# ═══════════════════════════════════════════════
#  ТОЧКА ВХОДА
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Brain Server v0.5')
    p.add_argument('--inputs',  type=int, default=6)
    p.add_argument('--outputs', type=int, default=3)
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