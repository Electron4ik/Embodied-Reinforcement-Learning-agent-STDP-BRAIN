import random
import sys
import json
import socket
import pygame


# =========================================================
# BrainClient — встроен прямо сюда, чтобы всё было в 1 файле
# =========================================================

class BrainClient:
    def __init__(self, host='127.0.0.1', port=7777, timeout=5.0):
        self.host = host
        self.port = port
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self.buf = b''

    def close(self):
        try:
            self.sock.close()
        except Exception:
            pass

    def _send(self, msg: dict) -> dict:
        data = (json.dumps(msg) + '\n').encode('utf-8')
        self.sock.sendall(data)
        return self._recv()

    def _recv(self) -> dict:
        while b'\n' not in self.buf:
            chunk = self.sock.recv(4096)
            if not chunk:
                raise ConnectionError("Brain Server closed connection")
            self.buf += chunk

        line, self.buf = self.buf.split(b'\n', 1)
        line = line.strip()
        if not line:
            return self._recv()

        resp = json.loads(line.decode('utf-8'))
        if resp.get('type') == 'error':
            raise RuntimeError(f"Brain Server: {resp['msg']}")
        return resp

    def step(self, inputs):
        resp = self._send({'type': 'step', 'inputs': list(inputs)})
        return int(resp['output'])

    def reward(self, value: float):
        self._send({'type': 'reward', 'value': float(value)})

    def reset(self):
        self._send({'type': 'reset'})

    def save(self, path: str):
        self._send({'type': 'save', 'path': path})

    def load(self, path: str):
        self._send({'type': 'load', 'path': path})

    def info(self):
        return self._send({'type': 'info'})

# =========================================================
# Игра
# =========================================================

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def main():
    print("Подключение к Brain Server...")
    try:
        brain = BrainClient(host='127.0.0.1', port=7777)
        info = brain.info()
        print(f"Мозг подключен: {info['n_inputs']} входов -> {info['n_outputs']} выходов.")
    except Exception as e:
        print("Ошибка: Brain Server не запущен или не отвечает.")
        print("Выполните: python brain_server.py --inputs 5 --outputs 3")
        print(f"Деталь: {e}")
        sys.exit(1)

    pygame.init()
    WIDTH, HEIGHT = 1200, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT), vsync=1)
    clock = pygame.time.Clock()
    running = True

    player_pos = pygame.Vector2(WIDTH / 2, HEIGHT / 2)
    ball_pos = pygame.Vector2(WIDTH / 2, 80)
    ball_velocity = 70.0 * 2
    g = 9.8 * 100 * 2

    player_speed = 700.0 * 2
    paddle_half_w = 100.0
    paddle_h = 50.0

    episode = 1
    score = 0
    prev_dist_x = abs(player_pos.x - ball_pos.x)

    accumulated_reward = 0.0
    save_every = 10

    # если хочешь чуть больше исследовательского хаоса — поставь 0.05
    idle_penalty = 0.14
    miss_penalty = 1.0
    catch_bonus = 0.0
    dist_bonus_scale = 1.0

    try:
        while running:
            dt = clock.tick(75) / 1000.0
            if dt <= 0.0:
                dt = 1 / 75.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill("black")

            # ---- наблюдения ----
            dx = (ball_pos.x - player_pos.x) / WIDTH
            dy = ball_pos.y / HEIGHT
            ball_v = ball_velocity / 1500.0
            player_x_norm = player_pos.x / WIDTH
            dist_x = abs(player_pos.x - ball_pos.x)

            # НОВАЯ ФИШКА: Датчик габаритов (1.0 если мяч в пределах краев, иначе 0.0)
            # Сужаем зону срабатывания чуть меньше реальной ширины (0.8), 
            # чтобы он не ловил мяч самыми уголками, где физика может подвести.
            safe_zone = paddle_half_w * 0.8
            is_inside_bounds = 1.0 if dist_x < safe_zone else 0.0

            obs = [
                player_x_norm,          # 0: где платформа
                dx,                     # 1: направление и дистанция до мяча
                dy,                     # 2: высота мяча
                is_inside_bounds,       # 4: ДАТЧИК КРАЁВ (горит, если мы под мячом)
            ]

            # ---- запрос действия ----
            action = brain.step(obs)

            # action: 0 = влево, 1 = вправо, 2 = стоять
            if action == 0:
                player_pos.x -= player_speed * dt
            elif action == 1:
                player_pos.x += player_speed * dt

            player_pos.x = clamp(player_pos.x, paddle_half_w, WIDTH - paddle_half_w)

            # ---- физика мяча ----
            ball_velocity += g * dt
            ball_pos.y += ball_velocity * dt

            # ---- reward ----
            # reward за приближение к мячу по X — только ОДИН раз, без дубляжа
            # ---- reward (Твоя философия: Удовольствие и Боль) ----
            dist_after_move = abs(player_pos.x - ball_pos.x)
            reward = 0.0

            if dist_after_move <= paddle_half_w:
                # Правило 1: Удовольствие (Мяч над нами)
                # Считаем близость к центру: от 0.0 (на самом краю) до 1.0 (идеально по центру)
                accuracy = 1.0 - (dist_after_move / paddle_half_w)
                # Получаем кайф каждый кадр. Чем точнее стоим, тем больше дофамина.
                # Берем небольшое число (0.1), так как это суммируется ~75 раз в секунду
                reward += 0.1 * accuracy 
            else:
                # Правило 2: Боль (Мяч не над нами)
                # Считаем, насколько сильно мы промахнулись мимо габаритов
                distance_outside = (dist_after_move - paddle_half_w) / WIDTH
                # Бьем током каждый кадр. Чем дальше мы от мяча, тем сильнее разряд.
                reward -= 0.1 * distance_outside
            
            # Старые штрафы за idle и progress нам больше не нужны!
            # Сама "боль" заставит его бежать к мячу, а "удовольствие" - замереть ровно по центру.

            done = False
            paddle_y = player_pos.y + 250

            hit = (paddle_y - 25 < ball_pos.y < paddle_y + 25) and (dist_after_move < paddle_half_w)

            if hit:
                # accuracy = 1.0 - clamp(dist_after_move / paddle_half_w, 0.0, 1.0)
                # reward += catch_bonus * max(0.1, accuracy)
                score += 1
                done = True
                print(f"Эпизод {episode}: Поймал! Счёт: {score} (Точность: {accuracy:.2f})")

            elif ball_pos.y > HEIGHT - 25:
                # reward -= miss_penalty
                done = True
                print(f"Эпизод {episode}: Уронил.")
                score = 0

            # ---- отдать reward мозгу ----
            accumulated_reward += reward
            brain.reward(accumulated_reward)
            accumulated_reward = 0.0

            # ---- сброс эпизода ----
            if done:
                brain.reset()
                ball_velocity = 70.0
                ball_pos.y = 80
                ball_pos.x = random.randint(300, WIDTH - 300)
                # более честный стартовый разброс:

                prev_dist_x = abs(player_pos.x - ball_pos.x)
                episode += 1

                if episode % save_every == 0:
                    brain.save('brain_catcher.pkl')

            # ---- рисование ----
            paddle_y = player_pos.y + 250
            pygame.draw.rect(
                screen,
                "white",
                pygame.Rect(player_pos.x - paddle_half_w, paddle_y, paddle_half_w * 2, paddle_h),
                width=5
            )
            pygame.draw.circle(screen, "yellow", (int(ball_pos.x), int(ball_pos.y)), 25)

            # HUD
            font = pygame.font.SysFont("consolas", 22)
            texts = [
                f"Episode: {episode}",
                f"Score: {score}",
                f"Action: {action}",
                f"Reward: {reward:.3f}",
                f"dx: {dx:.3f}  ball_v: {ball_v:.3f}",
            ]
            for i, t in enumerate(texts):
                surf = font.render(t, True, (180, 180, 180))
                screen.blit(surf, (20, 20 + i * 24))

            pygame.display.flip()
            prev_dist_x = abs(player_pos.x - ball_pos.x)

    except KeyboardInterrupt:
        print("\nОстановка.")
    finally:
        try:
            brain.save('brain_catcher_final.pkl')
        except Exception:
            pass
        brain.close()
        pygame.quit()


if __name__ == "__main__":
    main()