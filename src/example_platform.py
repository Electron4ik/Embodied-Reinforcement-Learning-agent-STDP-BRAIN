import sys, json, socket, math
import pygame
import numpy as np

# ── BrainClient ──────────────────────────────────────
class BrainClient:
    def __init__(self, host='127.0.0.1', port=7777, timeout=5.0):
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
        r = self._send({'type': 'step', 'inputs': list(obs)})
        return r['output'], r['confidence']

    def reward(self, v):   self._send({'type': 'reward', 'value': float(v)})
    def reset(self):       self._send({'type': 'reset'})
    def save(self, p):     self._send({'type': 'save', 'path': p})
    def info(self):        return self._send({'type': 'info'})


# ── Физика ───────────────────────────────────────────
class Ball:
    GRAVITY = 900.0
    def __init__(self, x, vy_init=80.0):
        self.x  = float(x)
        self.y  = 50.0
        self.vx = 0.0 # Можно добавить рандом для сложности позже
        self.vy = vy_init

    def update(self, dt):
        self.vy += self.GRAVITY * dt
        self.x  += self.vx * dt
        self.y  += self.vy * dt


# ── Главный цикл ─────────────────────────────────────
def main():
    try:
        brain = BrainClient()
        info  = brain.info()
    except Exception as e:
        print(f"Ошибка Brain Server: {e}")
        sys.exit(1)

    pygame.init()
    W, H = 800, 750
    screen = pygame.display.set_mode((W, H), vsync=1)
    pygame.display.set_caption("Catcher v4 — Target Logic")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 18)

    PADDLE_HW = 90.0
    PADDLE_Y  = H - 70.0
    SPEED     = 700.0 # Чуть быстрее, чтобы успевал
    BALL_R    = 20

    def new_ball(ep):
        x = 120 + ((ep * 173) % (W - 240))
        return Ball(x)

    player_x = W / 2.0
    episode  = 0
    score, streak, best = 0, 0, 0
    ball = new_ball(episode)

    # Параметры Frame Skip
    skip_frames = 4  
    frame_counter = 0
    last_action = 0
    last_conf = 0.0

    try:
        while True:
            dt = min(clock.tick(60) / 1000.0, 0.033)
            frame_counter += 1

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: return

            # 1. Считаем предсказание точки падения
            dy_left = max(PADDLE_Y - ball.y, 1.0)
            t_impact = dy_left / max(ball.vy, 10.0)
            pred_x = ball.x + ball.vx * t_impact

            # 2. Логика нейронки (раз в N кадров)
            if frame_counter >= skip_frames:
                rel_x = (ball.x - player_x) / W
                dist_norm = abs(rel_x)
                ball_vx = float(np.clip(ball.vx / 600.0, -1, 1))
                ball_vy = float(np.clip(ball.vy / 1800.0, 0, 1))
                t_norm = float(np.clip(t_impact / 3.0, 0, 1))
                pred_rel_x = float(np.clip((pred_x - player_x) / W, -1, 1))

                obs = [rel_x, ball_vx, ball_vy, t_norm, pred_rel_x, dist_norm]
                last_action, last_conf = brain.step(obs)
                frame_counter = 0

            # 3. Движение
            old_player_x = player_x
            if last_action == 0: player_x -= SPEED * dt
            else: player_x += SPEED * dt
            player_x = float(np.clip(player_x, PADDLE_HW, W - PADDLE_HW))

            # 4. Расчет награды за сближение с ТОЧКОЙ УДАРА
            old_error = abs(pred_x - old_player_x)
            new_error = abs(pred_x - player_x)
            # Награда за каждый шаг в правильном направлении
            step_reward = (old_error - new_error) / W * 5.0
            
            # Микро-штраф за стояние у стены, только если он пытается ехать сквозь неё
            if old_player_x == player_x and (player_x <= PADDLE_HW or player_x >= W - PADDLE_HW):
                step_reward -= 0.005

            ball.update(dt)
            
            done = False
            total_reward = step_reward

            # Проверка финала эпизода
            if (PADDLE_Y - 15 < ball.y < PADDLE_Y + 20 and abs(ball.x - player_x) < PADDLE_HW):
                acc = 1.0 - (abs(ball.x - player_x) / PADDLE_HW)
                total_reward += 15.0 * max(0.1, acc) # Жирный бонус за центр платформы
                streak += 1
                score += 1
                best = max(best, streak)
                done = True
                print(f"Ep {episode:5d} | ✓ Catch! Streak: {streak} Acc: {acc:.2f}")
            
            elif ball.y > H + 30:
                total_reward -= 5.0 # "Не смертельный" штраф
                streak = 0
                score = 0
                done = True
                print(f"Ep {episode:5d} | ✗ Miss... Best: {best}")

            brain.reward(total_reward)

            if done:
                episode += 1
                brain.reset()
                ball = new_ball(episode)
                if episode % 50 == 0:
                    brain.save(f'brain_catcher_ep{episode}.pkl')
                    info = brain.info()

            # ── Рендер ──────────────────────────────────
            screen.fill((10, 12, 20))
            # Линия до цели
            pygame.draw.line(screen, (50, 50, 80), (int(ball.x), int(ball.y)), (int(pred_x), int(PADDLE_Y)), 1)
            pygame.draw.circle(screen, (255, 200, 0), (int(ball.x), int(ball.y)), BALL_R)
            
            # Платформа
            rect_color = (100, 255, 100) if streak > 0 else (200, 200, 200)
            pygame.draw.rect(screen, rect_color, (int(player_x - PADDLE_HW), int(PADDLE_Y), int(PADDLE_HW*2), 15), 2)

            # Текст
            status = [
                f"Episode: {episode} | Streak: {streak} | Best: {best}",
                f"Input: {last_action} ({'Left' if last_action==0 else 'Right'}) Conf: {last_conf:.2f}",
                f"Reward: {total_reward:+.4f}"
            ]
            for i, line in enumerate(status):
                screen.blit(font.render(line, True, (150, 150, 170)), (10, 10 + i*22))

            pygame.display.flip()

    except Exception as e: print(f"Критическая ошибка: {e}")
    finally:
        brain.close()
        pygame.quit()

if __name__ == '__main__': main()