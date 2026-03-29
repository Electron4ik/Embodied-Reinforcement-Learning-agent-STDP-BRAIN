"""
Brain Server v0.3
=================
Спайковый мозг как TCP сервис.

Новое в v0.3:
  - Eligibility traces — мозг помнит что привело к результату
  - Per-action value baseline — стабильное TD обучение
  - Weight decay — убирает мусор, даёт долгую память
  - Confidence freeze — не ломает то что работает
  - Energy система — внутренняя мотивация

Запуск:
    python brain_server.py --inputs 4 --outputs 2

Протокол (JSON строки, \\n разделитель):
    Клиент → {"type": "step",   "inputs": [0.1, -0.5, 0.03, 0.8]}
    Сервер → {"type": "action", "output": 1, "confidence": 0.72}
    Клиент → {"type": "reward", "value": 1.0}
    Клиент → {"type": "reset"}
    Клиент → {"type": "save",   "path": "brain.pkl"}
    Клиент → {"type": "load",   "path": "brain.pkl"}
    Клиент → {"type": "info"}
    Сервер → {"type": "ok"} | {"type": "error", "msg": "..."}
"""

import argparse
import json
import pickle
import socket
import threading
import numpy as np
from collections import deque


# ═══════════════════════════════════════════════════════
#  ПАРАМЕТРЫ СЕТИ
# ═══════════════════════════════════════════════════════

DT              = 0.25    # шаг симуляции, мс
TAU_M           = 15.0   # постоянная утечки мембраны
V_THRESH        = 1.0
V_RESET         = 0.0
REFRAC          = 5
NOISE_AMP       = 0.05

INPUT_RATE_HI   = 0.20
INPUT_RATE_LO   = 0.005

NEURONS_PER_IN  = 5
NEURONS_PER_OUT = 10
INTERNAL_MULT   = 8

# ── Обучение ─────────────────────────────────────────
LR              = 0.02
LR_STDP         = 0.008
LR_CRITIC       = 0.01   # state critic learning rate
TAU_PRE         = 20.0
TAU_ELIG        = 2250.0
TAU_MOD         = 500.0 # модуляторы затухают медленнее, чем eligibility, чтобы сохранять мотивацию между эпизодами
D_REWARD        = 2.0
P_PUNISH        = 0.7
W_MAX           = 4.0
W_MIN_IO        = 0.0
GAMMA           = 0.98   # discount factor для TD critic
# W_DECAY убран — он стирал выученные паттерны быстрее чем они закреплялись

TICKS_PER_STEP  = 80

# ── Energy ───────────────────────────────────────────
ENERGY_GAIN     = 0.01


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
        nin     = max(160, ni * INTERNAL_MULT)
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

        # Входы
        self.input_rates = np.full(ni, INPUT_RATE_LO)
        self.in_accum    = np.zeros(ni)
        self.out_counts  = np.zeros(no)

        # Traces
        self.pre_trace = np.zeros(self.NT)

        # Eligibility traces — (n_outputs, ni+nin)
        # E_out[i] = след активности который привёл к выбору класса i
        # Затухает между step()-вызовами, накапливается в течение эпизода
        self.E_out   = np.zeros((n_outputs, ni + nin))
        self._k_elig = np.exp(-DT * TICKS_PER_STEP / TAU_ELIG)

        # State critic — линейный оценщик ценности состояния
        # Вместо per-action baseline использует реальные признаки состояния
        # feat = [in_act, int_act] — вектор длиной NI+NIN
        self.critic_w  = np.zeros(ni + nin)
        self.last_feat = np.zeros(ni + nin)
        self.last_v    = 0.0

        # Модуляторы
        self.dopamine = 0.0
        self.pain     = 0.0
        self._k_mod   = np.exp(-DT / TAU_MOD)
        self._k_pre   = np.exp(-DT / TAU_PRE)

        # Energy
        self.energy = 1.0

        # Стат
        self.total_steps   = 0
        self.total_rewards = 0.0
        self._last_winner  = 0
        self._last_conf    = 0.0

    def _init_weights(self, rng):
        NI, NIN, NO = self.NI, self.NIN, self.NO
        OUT_S = self.OUT_START
        INH_S = self.INH_START
        INH_E = self.INH_END
        W     = np.zeros((self.NT, self.NT))

        m = rng.random((NIN, NI)) < 0.25
        W[NI:NI+NIN, :NI] = rng.uniform(0.10, 0.35, (NIN, NI)) * m

        m = rng.random((NIN, NIN)) < 0.05
        np.fill_diagonal(m, False)
        W[NI:NI+NIN, NI:NI+NIN] = rng.uniform(0.01, 0.08, (NIN, NIN)) * m

        m = rng.random((NIN, self.NIN_INH)) < 0.20
        W[NI:NI+NIN, INH_S:INH_E] = -rng.uniform(0.10, 0.30, (NIN, self.NIN_INH)) * m

        m = rng.random((NO, NI)) < 0.45
        W[OUT_S:, :NI] = rng.uniform(0.05, 0.20, (NO, NI)) * m

        m = rng.random((NO, NIN)) < 0.10
        W[OUT_S:, NI:NI+NIN] = rng.uniform(0.02, 0.10, (NO, NIN)) * m

        lat = 0.30
        for i in range(self.n_outputs):
            for j in range(self.n_outputs):
                if i == j: continue
                gi_s = OUT_S + i * NEURONS_PER_OUT
                gj_s = OUT_S + j * NEURONS_PER_OUT
                W[gj_s:gj_s+NEURONS_PER_OUT, gi_s:gi_s+NEURONS_PER_OUT] = -lat

        np.fill_diagonal(W, 0)
        return W

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

        # STDP internal×internal
        mod = (self.dopamine - self.pain) * (0.5 + self.energy)
        if abs(mod) > 0.05:
            np.outer(self.spikes_f[int_s], self.pre_trace[int_s], out=self._dW_buf)
            self._dW_buf *= LR_STDP * mod
            self._dW_buf *= self.conn[int_s, int_s]
            self.W[int_s, int_s] += self._dW_buf
            self.W[int_s, INH_S:INH_E] = -np.abs(self.W[int_s, INH_S:INH_E])
            self.W[int_s, int_s] *= 0.99995
            np.clip(self.W[int_s, int_s], -W_MAX, W_MAX, out=self.W[int_s, int_s])

        self.dopamine *= self._k_mod
        self.pain     *= self._k_mod

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

        action, confidence = self._read_output()

        # Eligibility: затухание + добавление текущей активности
        # Взвешиваем на (0.5 + confidence) — уверенный выбор оставляет более сильный след
        self.E_out *= self._k_elig
        in_act  = self.in_accum / max(self.in_accum.max(), 1.0)
        int_act = self.pre_trace[self.NI:self.NI+self.NIN]
        int_act = int_act / max(int_act.max(), 1.0)
        feat    = np.concatenate([in_act, int_act])
        self.E_out[action] += feat * (0.5 + confidence)

        # Critic: запоминаем признаки и оценку состояния для TD update в reward()
        self.last_feat = feat.copy()
        self.last_v    = float(self.critic_w @ feat)

        self.total_steps += 1
        self._last_conf   = confidence
        return action, confidence

    def reward(self, value: float):
        """
        Actor-Critic TD обучение с асимметричным клипингом.
        Позитивные ошибки усиливают агрессивнее, негативные — мягче.
        Это предотвращает разрушение выученного после одной ошибки.
        """
        self.total_rewards += value
        self.energy = float(np.clip(self.energy + ENERGY_GAIN * value, 0.0, 1.0))

        td_error = value - self.last_v

        # Асимметричный клипинг: успех может быть большим, провал — ограничен
        # Без этого один мисс после серии поймок разносит всю стратегию
        td_error = float(np.clip(td_error, -0.8, 2.0))

        # Critic обновляется только на значимых ошибках
        if abs(td_error) > 0.01:
            self.critic_w += LR_CRITIC * td_error * self.last_feat

        if td_error > 0:
            self.dopamine += D_REWARD * min(td_error, 2.0)
        else:
            self.pain += P_PUNISH * min(-td_error, 2.0)

        self._apply_reward_elig(td_error)

    def reset(self):
        """Сброс нейронного состояния. Веса живут дальше. E_out частично сохраняется."""
        self.V[:]          = 0.0
        self.refrac[:]     = 0
        self.trace[:]      = 0.0
        self.pre_trace[:]  = 0.0
        self.spikes[:]     = False
        self.spikes_f[:]   = 0.0
        self.out_counts[:] = 0.0
        self.in_accum[:]   = 0.0
        # E_out не сбрасываем полностью — мозг помнит что работало между эпизодами
        # Но приглушаем на 30% чтобы старые ошибки не накапливались бесконечно
        self.E_out *= 0.70

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({
                'n_inputs':  self.n_inputs,
                'n_outputs': self.n_outputs,
                'seed':      self.seed,
                'W':         self.W,
                'critic_w':  self.critic_w,
                'E_out':     self.E_out,
                'stats': {'total_steps': self.total_steps,
                          'total_rewards': self.total_rewards}
            }, f)

    @classmethod
    def load(cls, path: str) -> 'Brain':
        with open(path, 'rb') as f:
            data = pickle.load(f)
        b = cls(data['n_inputs'], data['n_outputs'], data['seed'])
        b.W          = data['W']
        b.conn       = b.W != 0
        b.critic_w   = data.get('critic_w', b.critic_w)
        b.E_out      = data.get('E_out', b.E_out)
        b.total_steps   = data['stats']['total_steps']
        b.total_rewards = data['stats']['total_rewards']
        return b

    def _read_output(self) -> tuple:
        counts = self.out_counts.reshape(self.n_outputs, NEURONS_PER_OUT).sum(axis=1)
        total  = counts.sum()

        if total == 0:
            w = int(np.random.randint(0, self.n_outputs))
            self._last_winner = w
            return w, 0.0

        # Stochastic action selection — softmax с температурой
        # Чем меньше уверенность в прошлый раз — тем больше исследование
        temp = max(0.15, 1.0 - self._last_conf)
        logits = counts / (temp + 1e-8)
        logits -= logits.max()          # стабилизация softmax
        probs  = np.exp(logits)
        probs /= probs.sum()
        w  = int(np.random.choice(self.n_outputs, p=probs))
        cf = float(counts[w] / total)
        self._last_winner = w
        return w, cf

    def _apply_reward_elig(self, prediction_error: float):
        """
        Обновление весов через eligibility traces.

        Асимметрия: негативные апдейты слабее позитивных.
        Confidence freeze: уверенный правильный ответ меняется слабее.
        """
        OUT_S  = self.OUT_START
        NI     = self.NI
        NIN    = self.NIN
        winner = self._last_winner

        magnitude = abs(prediction_error)

        # 1. Защита от "переучивания" (Confidence Freeze)
        if prediction_error > 0 and self._last_conf > 0.85:
            magnitude *= 0.2  # Если и так всё хорошо, почти не трогаем веса
        
        # 2. Асимметрия: разрушать сложнее, чем строить
        if prediction_error < 0:
            magnitude *= 0.4  # Штраф мягче, чтобы не стереть память за один раз

        if magnitude < 1e-5: return

        # Подготовка дельты и шума для борьбы с клонированием
        # (Создаем матрицу шума под размер группы нейронов выхода)
        noise = np.random.uniform(0.7, 1.3, size=(NEURONS_PER_OUT, NI + NIN))
        delta = LR * magnitude * self.E_out[winner][None, :] * noise

        w_s = OUT_S + winner * NEURONS_PER_OUT
        w_e = w_s + NEURONS_PER_OUT

        if prediction_error > 0:
            # Усиливаем победителя
            self.W[w_s:w_e, :NI+NIN] += delta
            # Ослабляем проигравших (контрастное обучение)
            for other in range(self.n_outputs):
                if other == winner: continue
                o_s = OUT_S + other * NEURONS_PER_OUT
                o_e = o_s + NEURONS_PER_OUT
                self.W[o_s:o_e, :NI+NIN] -= delta * 0.3 
        else:
            # Наказываем за ошибку
            self.W[w_s:w_e, :NI+NIN] -= delta
            # Даем шанс другим (чуть-чуть подталкиваем их веса вверх)
            for other in range(self.n_outputs):
                if other == winner: continue
                o_s = OUT_S + other * NEURONS_PER_OUT
                o_e = o_s + NEURONS_PER_OUT
                self.W[o_s:o_e, :NI+NIN] += delta * 0.2

        # Ограничиваем веса, чтобы мозг не "взорвался"
        np.clip(self.W[OUT_S:, :NI], W_MIN_IO, W_MAX, out=self.W[OUT_S:, :NI])
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
            'energy':        round(self.energy, 3),
            'critic_v':      round(self.last_v, 3),
        }


# ═══════════════════════════════════════════════════════
#  TCP СЕРВЕР
# ═══════════════════════════════════════════════════════

class BrainServer:
    def __init__(self, brain: Brain, host: str = '127.0.0.1', port: int = 7777):
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
        print(f"[Brain v0.3] {self.host}:{self.port}")
        print(f"[Brain] {self.brain.n_inputs} входов → {self.brain.n_outputs} выходов")
        print(f"[Brain] {self.brain.NT} нейронов, {np.count_nonzero(self.brain.W)} синапсов")
        print(f"[Brain] Eligibility TAU={TAU_ELIG}ms  LR={LR}  Critic LR={LR_CRITIC}\n")
        try:
            while self._running:
                srv.settimeout(1.0)
                try:
                    conn, addr = srv.accept()
                except socket.timeout:
                    continue
                print(f"[Brain] + {addr[0]}:{addr[1]}")
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
            print(f"[Brain] - {addr[0]}:{addr[1]}")
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
            print(f"[Brain] Сохранено: {path}")
            return {'type': 'ok', 'path': path}

        elif t == 'load':
            path = msg['path']
            with self._lock:
                loaded = Brain.load(path)
                if loaded.n_inputs != self.brain.n_inputs or \
                   loaded.n_outputs != self.brain.n_outputs:
                    return {'type': 'error',
                            'msg': f'Несовместимо: {loaded.n_inputs}→{loaded.n_outputs}'}
                self.brain.W                = loaded.W
                self.brain.conn             = loaded.conn
                self.brain.critic_w = loaded.critic_w
                self.brain.E_out            = loaded.E_out
                self.brain.total_steps      = loaded.total_steps
                self.brain.total_rewards    = loaded.total_rewards
            print(f"[Brain] Загружено: {path}")
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
    p.add_argument('--inputs',  type=int, default=4)
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