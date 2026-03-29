"""
example_cartpole.py
===================
Демо среда — маятник (CartPole). Подключается к Brain Server.

Запуск (в двух терминалах):
    Terminal 1:  python brain_server.py --inputs 4 --outputs 2
    Terminal 2:  python example_cartpole.py

Мозг не знает что снаружи маятник.
Маятник не знает что внутри нейроны.
"""

import math
import time
import sys
from brain_client import BrainClient

# ── Физика маятника ───────────────────────────────────

GRAVITY     = 9.8
CART_MASS   = 1.0
POLE_MASS   = 0.1
POLE_HALF_L = 0.5
FORCE       = 10.0
DT          = 0.02    # секунды

MAX_ANGLE   = math.radians(12)   # падение при > 12°
MAX_POS     = 2.4                # вылет за край

class CartPole:
    def __init__(self):
        self.reset()

    def reset(self):
        import random
        self.x     = random.uniform(-0.05, 0.05)
        self.x_dot = random.uniform(-0.05, 0.05)
        self.theta     = random.uniform(-0.05, 0.05)
        self.theta_dot = random.uniform(-0.05, 0.05)
        self.steps = 0
        return self._obs()

    def _obs(self):
        """Нормализованные наблюдения [-1, 1] для мозга."""
        return [
            self.x      / MAX_POS,
            self.x_dot  / 5.0,
            self.theta  / MAX_ANGLE,
            self.theta_dot / 5.0,
        ]

    def step(self, action: int):
        """action: 0 = влево, 1 = вправо"""
        force = FORCE if action == 1 else -FORCE

        total_mass    = CART_MASS + POLE_MASS
        pm_l          = POLE_MASS * POLE_HALF_L
        cos_th        = math.cos(self.theta)
        sin_th        = math.sin(self.theta)

        tmp = (force + pm_l * self.theta_dot**2 * sin_th) / total_mass
        theta_acc = (GRAVITY * sin_th - cos_th * tmp) / \
                    (POLE_HALF_L * (4/3 - POLE_MASS * cos_th**2 / total_mass))
        x_acc = tmp - pm_l * theta_acc * cos_th / total_mass

        self.x         += DT * self.x_dot
        self.x_dot     += DT * x_acc
        self.theta     += DT * self.theta_dot
        self.theta_dot += DT * theta_acc
        self.steps     += 1

        done = (abs(self.x)     > MAX_POS or
                abs(self.theta) > MAX_ANGLE)

        # Dense reward — мозг видит направление к улучшению, не только жив/мертв
        reward = (
            1.0
            - 0.8 * abs(self.theta)    / MAX_ANGLE
            - 0.2 * abs(self.x)        / MAX_POS
            - 0.05 * abs(self.theta_dot) / 5.0
            - 0.02 * abs(self.x_dot)   / 5.0
        )
        if done:
            reward -= 5.0

        return self._obs(), reward, done


# ── Главный цикл ──────────────────────────────────────

def main():
    print("Подключаюсь к Brain Server...")
    try:
        brain = BrainClient(host='127.0.0.1', port=7777)
    except ConnectionRefusedError:
        print("Brain Server не запущен!")
        print("  python brain_server.py --inputs 4 --outputs 2")
        sys.exit(1)

    info = brain.info()
    print(f"Подключено. Мозг: {info['n_inputs']} входов → {info['n_outputs']} выходов, "
          f"{info['n_neurons']} нейронов\n")

    env          = CartPole()
    episode      = 0
    best_score   = 0
    scores       = []

    try:
        while True:
            obs   = env.reset()
            brain.reset()
            score = 0

            while True:
                # Мозг решает что делать
                action = brain.step(obs)

                # Среда реагирует
                obs, reward, done = env.step(action)
                score += 1

                # Мозг получает результат
                brain.reward(reward)

                if done:
                    break

            episode += 1
            scores.append(score)
            best_score = max(best_score, score)
            avg = sum(scores[-20:]) / min(len(scores), 20)

            bar_len = min(score, 60)
            bar     = '█' * bar_len + '░' * (60 - bar_len)
            status  = "🏆" if score == best_score and episode > 1 else "  "

            print(f"Ep {episode:4d} | [{bar}] {score:4d} шагов | "
                  f"avg20={avg:5.1f} | best={best_score} {status}")

            # Каждые 50 эпизодов — сохранение
            if episode % 50 == 0:
                brain.save(f'brain_cartpole_ep{episode}.pkl')
                print(f"         → сохранено: brain_cartpole_ep{episode}.pkl")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nОстановка.")
        brain.save('brain_cartpole_final.pkl')
        print(f"Финальное состояние сохранено.  Лучший результат: {best_score} шагов.")
    finally:
        brain.close()


if __name__ == '__main__':
    main()