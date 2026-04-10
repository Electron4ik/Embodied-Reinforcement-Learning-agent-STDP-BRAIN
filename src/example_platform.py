import sys, json, socket, gif_pygame, math
import pygame
import numpy as np

gif = gif_pygame.load("brain.gif") 

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
        self.vx = 0.0 
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
    pygame.display.set_caption("RL Catcher | Synchronized FrameSkip")
    clock  = pygame.time.Clock()
    font   = pygame.font.SysFont("consolas", 18)

    PADDLE_HW = 90.0
    PADDLE_H  = 16.0
    PADDLE_Y  = H - 70.0
    SPEED     = 700.0 
    BALL_R    = 20
    FRAME_SKIP = 5 # Каждые 5 кадров запрашиваем новое решение

    def new_ball(ep):
        # Мяч всегда в центре для теста "обучаемости покою"
        return Ball(W / 2.0)

    player_x = W / 2.0
    episode  = 0
    score, streak, best = 0, 0, 0
    ball = new_ball(episode)
    
    # Состояние для синхронизации
    frames_passed = 0
    accumulated_reward = 0.0
    last_action = 2 # По умолчанию стоим
    last_conf = 0.0

    try:
        while True:
            dt = min(clock.tick(60) / 1000.0, 0.033)

            for ev in pygame.event.get():
                if ev.type == pygame.QUIT: return

            # 1. Предсказание точки падения
            dy_left = max(PADDLE_Y - ball.y, 1.0)
            t_impact = dy_left / max(ball.vy, 10.0)
            pred_x = ball.x + ball.vx * t_impact
            target_x = float(np.clip(pred_x, PADDLE_HW, W - PADDLE_HW))

            # 2. Формируем наблюдения (State)
            rel_x = (ball.x - player_x) / W
            dist_norm = abs(rel_x)
            ball_vx = float(np.clip(ball.vx / 600.0, -1, 1))
            ball_vy = float(np.clip(ball.vy / 1800.0, 0, 1))
            t_norm = float(np.clip(t_impact / 3.0, 0, 1))
            pred_rel_x = float(np.clip((pred_x - player_x) / W, -1, 1))
            obs = [rel_x, ball_vx, ball_vy, t_norm, pred_rel_x, dist_norm]

            # 3. Синхронное взаимодействие с мозгом (Frame Skip)
            if frames_passed == 0:
                # Отправляем накопленную за прошлые 5 кадров награду ПЕРЕД новым шагом
                # (Если это самый первый кадр игры, награда 0)
                brain.reward(accumulated_reward)
                accumulated_reward = 0.0
                
                # Запрашиваем новое действие
                last_action, last_conf = brain.step(obs)

            # 4. Движение (согласно последнему выбранному действию)
            old_player_x = player_x
            if last_action == 0: player_x -= SPEED * dt # Влево
            elif last_action == 1: player_x += SPEED * dt # Вправо
            # action == 2 -> ничего не делаем (стоим)
            
            player_x = float(np.clip(player_x, PADDLE_HW, W - PADDLE_HW))

            # 5. Позиционная награда за кадр
            dist_to_target = abs(target_x - player_x)
            # Награда за близость к центру падения
            # step_rew = 0.05 * (1.0 - (dist_to_target / (W/2))) # сейчас награда даже не бывает отрицательной, что нужно сделать?
            step_rew = 0.1 * (1.0 - (dist_to_target / (W/2))) - 0.05 * (dist_to_target / (W/2)) # так награда будет отрицательной, если далеко от цели
            
            # Штраф за "залипание" в стену
            if old_player_x == player_x and (player_x <= PADDLE_HW or player_x >= W - PADDLE_HW):
                step_rew -= 0.02
            
            accumulated_reward += step_rew

            # 6. Физика
            ball.update(dt)
            
            done = False
            # Проверка: Поймал?
            if (PADDLE_Y - 15 < ball.y < PADDLE_Y + PADDLE_H + 5 and abs(ball.x - player_x) < PADDLE_HW):
                acc = 1.0 - (abs(ball.x - player_x) / PADDLE_HW)
                accumulated_reward += 5.0 * max(0.2, acc) 
                streak += 1
                best = max(best, streak)
                done = True
                print(f"Ep {episode:5d} | ✓ Catch! Streak: {streak} Acc: {acc:.2f} Conf: {last_conf:.2f}")
            
            # Проверка: Уронил?
            elif ball.y > H + 30:
                accumulated_reward -= 1.2
                streak = 0
                done = True
                print(f"Ep {episode:5d} | ✗ Miss... Best: {best}")

            if done:
                # В конце эпизода ВСЕГДА шлем финальную награду
                brain.reward(accumulated_reward)
                accumulated_reward = 0.0
                frames_passed = 0
                episode += 1
                brain.reset()
                ball = new_ball(episode)
                # ball.x = W / 2.0 + np.random.uniform(-300, 300) # Немного рандома в начальной позиции, но нужно чтобы мяч не был в центре, а ближе к краям:
                ball.x = W / 2.0 + (PADDLE_HW + 50) * np.random.choice([-1, 1]) # Рандомно слева или справа от центра, но не в центре
                
                if episode % 50 == 0:
                    brain.save(f'brain_catcher_ep{episode}.pkl')
            else:
                # Считаем кадры до следующего запроса нейронки
                frames_passed = (frames_passed + 1) % FRAME_SKIP

            # ── Рендер ──────────────────────────────────
            # screen.fill((5, 15, 5) if streak > 0 else (20, 5, 5)) 
            screen.fill((0, 0, 0))
            gif.render(screen, ((W-gif.get_width())*0.5, (H-gif.get_height())*0.5))

            # Линия до предсказаынной цели
            pygame.draw.line(screen, (255, 255, 255), (int(ball.x), int(ball.y)), (int(target_x), int(PADDLE_Y)), 1)
            pygame.draw.circle(screen, (255, 200, 0), (int(ball.x), int(ball.y)), BALL_R)
            
            # Зона поимки (визуальная подсказка)

            pygame.draw.line(screen, (30, 40, 30), (int(target_x - PADDLE_HW*0.6), int(PADDLE_Y+5)),
                                                   (int(target_x + PADDLE_HW*0.6), int(PADDLE_Y+5)), 4)

            # Платформа
            rect_color = (100, 255, 100) if accumulated_reward > 0 else (255, 100, 100)
            pygame.draw.rect(screen, rect_color, (int(player_x - PADDLE_HW), int(PADDLE_Y), int(PADDLE_HW*2), int(PADDLE_H)), 2)

            # Отрисовка текста
            act_name = ["LEFT ", "RIGHT", "STAY "][last_action] if last_action < 3 else "???"
            status = [
                f"Episode: {episode} | Best: {best}",
                f"Action: {act_name} | Conf: {last_conf:.2f}",
                f"Streak: {streak} | FrameCounter: {frames_passed}",
                f"Reward: {accumulated_reward:.3f}"
            ]
            for i, line in enumerate(status):
                screen.blit(font.render(line, True, (200, 200, 200)), (10, 10 + i*22))

            pygame.display.flip()

    except Exception as e: 
        print(f"Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
    finally:
        brain.close()
        pygame.quit()

if __name__ == '__main__': main()