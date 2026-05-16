"""
hovercraft.py — 1D агент с инерцией (Strict Rules Version)
=====================================
Запуск:
    Terminal 1: python brain_server.py --inputs 6 --outputs 3
    Terminal 2: python hovercraft.py
"""

import sys, json, socket
import pygame
import numpy as np


# ── BrainClient ─────────────────────────────────
class BrainClient:
    def __init__(self, host='127.0.0.1', port=7777, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf  = b''

    def close(self):
        try: self.sock.close()
        except: pass

    def _send(self, msg):
        self.sock.sendall((json.dumps(msg) + '\n').encode())
        return self._recv()

    def _recv(self):
        while b'\n' not in self.buf:
            c = self.sock.recv(4096)
            if not c: raise ConnectionError("closed")
            self.buf += c
        line, self.buf = self.buf.split(b'\n', 1)
        r = json.loads(line.strip())
        if r.get('type') == 'error': raise RuntimeError(r['msg'])
        return r

    def step(self, obs):
        r = self._send({'type': 'step', 'inputs': [float(x) for x in obs]})
        return r['output'], r['confidence']

    def reward(self, v, next_obs=None, terminal=False):
        msg = {
            'type': 'reward',
            'value': float(v),
            'terminal': bool(terminal)
        }
        if next_obs is not None:
            msg['next_inputs'] = [float(x) for x in next_obs]
        self._send(msg)

    def reset(self):   self._send({'type': 'reset'})
    def save(self, p): self._send({'type': 'save', 'path': p})
    def info(self):    return self._send({'type': 'info'})


# ── Физика агента ────────────────────────────────
class Agent:
    THRUST      = 280.0   # ускорение от тяги
    FRICTION    = 0.88    # коэффициент трения (за кадр)
    MAX_SPEED   = 350.0

    def __init__(self, x, world_w):
        self.x  = float(x)
        self.vx = 0.0
        self.world_w = float(world_w)

    def update(self, action: int, dt: float) -> bool:
        """Возвращает True если врезался в стену."""
        if action == 0:
            self.vx -= self.THRUST * dt
        elif action == 1:
            self.vx += self.THRUST * dt

        self.vx *= (self.FRICTION ** dt)   # трение
        self.vx  = float(np.clip(self.vx, -self.MAX_SPEED, self.MAX_SPEED))
        self.x  += self.vx * dt

        if self.x <= 0 or self.x >= self.world_w:
            self.x = float(np.clip(self.x, 0, self.world_w))
            self.vx = 0.0
            return True
        return False

    def predict_pos(self, steps: int = 15, dt: float = 1/60) -> float:
        """Куда агент приедет если ничего не делать (коастинг)."""
        vx = self.vx
        x  = self.x
        for _ in range(steps):
            vx *= (self.FRICTION ** dt)
            x  += vx * dt
        return float(np.clip(x, 0, self.world_w))


# ── Главный цикл ─────────────────────────────────
def main():
    print("Подключение к Brain Server...")
    try:
        brain = BrainClient()
        info  = brain.info()
        print(f"Подключено: {info['n_inputs']} входов → {info['n_outputs']} выходов")
        if info['n_inputs'] != 6 or info['n_outputs'] != 3:
            print("ОШИБКА: нужно --inputs 6 --outputs 3")
            sys.exit(1)
    except Exception as e:
        print(f"Brain Server не запущен: {e}")
        print("  python brain_server.py --inputs 6 --outputs 3")
        sys.exit(1)

    pygame.init()
    W, H = 900, 300
    screen = pygame.display.set_mode((W, H), vsync=1)
    pygame.display.set_caption("Hovercraft — Brain v1.0 (Strict)")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 17)

    AGENT_Y   = H // 2
    AGENT_R   = 18
    TARGET_R  = 14
    TARGET_TOL = 30.0   # попал в цель если ближе этого

    MAX_STEPS_PER_TARGET = 350 # Увеличен лимит для неспешной езды

    def new_target():
        # Спавним достаточно далеко от краев, чтобы не было несправедливых смертей
        return float(np.random.uniform(80, W - 80))

    HOLD_FRAMES = 4
    hold_counter = 0

    agent    = Agent(W / 2, W)
    target_x = new_target()
    episode  = 0
    total_reaches = 0
    streak   = 0
    best     = 0

    step_in_ep   = 0
    prev_action  = 2
    acc_reward   = 0.0

    r_hist    = []

    try:
        while True:
            dt = min(clock.tick(60) / 1000.0, 0.033)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    return

            # ── Наблюдения ───────────────────────────────
            dist   = (target_x - agent.x) / W      # [-1..1]
            vel    = float(np.clip(agent.vx / Agent.MAX_SPEED, -1, 1))
            wall_l = agent.x / W
            wall_r = (W - agent.x) / W
            pred   = (agent.predict_pos() - target_x) / W
            pred   = float(np.clip(pred, -1, 1))
            bias   = 1.0

            obs = [dist, vel, wall_l, wall_r, pred, bias]

            # ── Мозг решает ──────────────────────────────
            action, conf = brain.step(obs)
            # 0=влево, 1=вправо, 2=стоять

            prev_dist = abs(target_x - agent.x)

            # ── Физика ───────────────────────────────────
            hit_wall = agent.update(action, dt)
            step_in_ep += 1

            curr_dist = abs(target_x - agent.x)

            # ── Формирование награды (Reward Shaping) ────
            r = 0.0

            # 1. Награда за сближение (Потенциальная)
            r += (prev_dist - curr_dist) * 4.0

            # 2. Награда за намерение ("Хлебные крошки")
            direction_to_target = 1 if target_x > agent.x else -1
            if (agent.vx * direction_to_target) > 10.0:  
                r += 0.005 # Крошечный бонус за правильное направление скорости

            # 3. Налоги и штрафы за энергозатраты
            r -= 0.001  # Базовый налог на время
            if action in (0, 1): r -= 0.002 # Налог на бензин
            if action != prev_action and prev_action != 2: r -= 0.01 # Налог на суету
            if action == 2 and abs(dist) > 0.15: r -= 0.015 # Налог на лень вдали от цели

            done     = False
            terminal = False

            # ── Проверка завершения эпизода ──────────────

            # Сценарий А: МГНОВЕННАЯ СМЕРТЬ ОБ СТЕНУ
            if hit_wall:
                r -= 2.0  # Жесткий штраф
                done = True
                terminal = True
                streak = 0
                print(f"Ep {episode:5d} | ☠️ СТЕНА! Score: {acc_reward:.2f}")

            # Сценарий Б: ТАЙМАУТ (Завис на месте)
            elif step_in_ep >= MAX_STEPS_PER_TARGET:
                r -= 1.0
                done = True
                terminal = True
                streak = 0
                print(f"Ep {episode:5d} | ⏱ ТАЙМАУТ. Score: {acc_reward:.2f}")

            # Сценарий В: ПОБЕДА (Достиг цели и затормозил)
            else:
                hit_now = abs(agent.x - target_x) < TARGET_TOL
                if hit_now:
                    hold_counter += 1
                    r += 0.05 # Греется в лучах славы
                else:
                    hold_counter = 0

                if hold_counter >= HOLD_FRAMES and abs(agent.vx) < 25.0:
                    r += 6.0 # Большой куш
                    total_reaches += 1
                    streak += 1
                    if streak > best: best = streak
                    done = True
                    terminal = True
                    print(f"Ep {episode:5d} | ★ ВЗЯЛ ЦЕЛЬ! Score: {acc_reward:.2f} | Streak: {streak}")

            acc_reward += r

            # Отправка следующего состояния в Critic
            next_dist = (target_x - agent.x) / W
            next_vel = float(np.clip(agent.vx / Agent.MAX_SPEED, -1, 1))
            next_wall_l = agent.x / W
            next_wall_r = (W - agent.x) / W
            next_pred = float(np.clip((agent.predict_pos() - target_x) / W, -1, 1))
            next_obs = [next_dist, next_vel, next_wall_l, next_wall_r, next_pred, 1.0]

            brain.reward(r, next_obs=next_obs, terminal=done)
            prev_action = action

            # ── Сброс среды ──────────────────────────────
            if done and terminal:
                r_hist.append(acc_reward)
                brain.reset()
                
                # Спавним агента в случайном месте (но не слишком близко к краям)
                agent      = Agent(np.random.uniform(100, W - 100), W)
                target_x   = new_target()
                episode   += 1
                step_in_ep = 0
                acc_reward = 0.0
                prev_action = 2
                hold_counter = 0

                if episode % 100 == 0:
                    brain.save(f'brain_hover_ep{episode}.pkl')
                    info = brain.info()

            # ── Рендер ───────────────────────────────────
            g = min(20 + streak * 8, 60)
            screen.fill((8, g, 12))

            pygame.draw.line(screen, (40, 50, 40), (0, AGENT_Y), (W, AGENT_Y), 1)

            pygame.draw.rect(screen, (30, 60, 30),
                             (int(target_x - TARGET_TOL), AGENT_Y - 8,
                              int(TARGET_TOL * 2), 16))
            pygame.draw.circle(screen, (80, 255, 80),
                               (int(target_x), AGENT_Y), TARGET_R, 2)

            pred_px = int(agent.predict_pos())
            pygame.draw.circle(screen, (60, 80, 160), (pred_px, AGENT_Y - 6), 5)

            a_color = {0: (100, 180, 255), 1: (255, 180, 100), 2: (200, 200, 200)}
            pygame.draw.circle(screen, a_color.get(action, (200,200,200)),
                               (int(agent.x), AGENT_Y), AGENT_R)
            
            vx_px = int(agent.vx / Agent.MAX_SPEED * 40)
            pygame.draw.line(screen, (255, 255, 100),
                             (int(agent.x), AGENT_Y),
                             (int(agent.x) + vx_px, AGENT_Y), 3)

            act_name = ["← thrust", "→ thrust", "· coast "][action]
            v_str    = f"{info.get('critic_v', 0.0):+.2f}"
            lines = [
                f"Ep {episode}  Reaches: {total_reaches}  Best streak: {best}",
                f"Action: {act_name}  conf={conf:.2f}  explore={info.get('explore',0):.3f}",
                f"dist={dist:+.3f}  vel={vel:+.3f}  pred={pred:+.3f}",
                f"Critic V={v_str}  step={step_in_ep}/{MAX_STEPS_PER_TARGET}",
                f"Ep Score: {acc_reward:.2f}",
            ]
            for i, t in enumerate(lines):
                screen.blit(font.render(t, True, (150, 160, 150)), (10, 8 + i*19))

            pygame.display.flip()

    except KeyboardInterrupt:
        print("\nОстановка.")
    finally:
        try: brain.save('brain_hover_final.pkl')
        except: pass
        brain.close()
        pygame.quit()

if __name__ == '__main__':
    main()