"""
brain_client.py
===============
Python-клиент для Brain Server. Подключается по TCP, прозрачный интерфейс.

Использование:
    from brain_client import BrainClient

    brain = BrainClient(host='127.0.0.1', port=7777)

    action = brain.step([0.1, -0.5, 0.03, 0.8])   # подать сенсоры
    brain.reward(+1.0)                              # дать награду
    brain.reset()                                   # сброс между эпизодами
    brain.save('checkpoint.pkl')
    info = brain.info()
    print(info)
"""

import json
import socket


class BrainClient:
    def __init__(self, host: str = '127.0.0.1', port: int = 7777, timeout: float = 10.0):
        self.host    = host
        self.port    = port
        self.timeout = timeout
        self._sock   = None
        self._buf    = b''
        self._connect()

    def _connect(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        self._buf = b''

    def _send(self, msg: dict) -> dict:
        data = (json.dumps(msg) + '\n').encode()
        self._sock.sendall(data)
        return self._recv()

    def _recv(self) -> dict:
        while b'\n' not in self._buf:
            chunk = self._sock.recv(4096)
            if not chunk:
                raise ConnectionError("Brain Server закрыл соединение")
            self._buf += chunk
        line, self._buf = self._buf.split(b'\n', 1)
        resp = json.loads(line.strip())
        if resp.get('type') == 'error':
            raise RuntimeError(f"Brain Server: {resp['msg']}")
        return resp

    # ── Публичный API ────────────────────────────────────

    def step(self, inputs: list) -> int:
        """
        Подать входные сигналы (числа от -1 до 1 или 0 до 1).
        Возвращает индекс выбранного действия (0..n_outputs-1).
        """
        resp = self._send({'type': 'step', 'inputs': list(inputs)})
        return resp['output']

    def step_full(self, inputs: list) -> tuple:
        """Как step(), но возвращает (action, confidence)."""
        resp = self._send({'type': 'step', 'inputs': list(inputs)})
        return resp['output'], resp['confidence']

    def reward(self, v, next_obs=None, terminal=False):
        msg = {
            'type': 'reward',
            'value': float(v),
            'terminal': bool(terminal)
        }
        if next_obs is not None:
            msg['next_inputs'] = [float(x) for x in next_obs]
        self._send(msg)

    def reset(self):
        """Сбросить состояние нейронов. Веса обучения сохраняются."""
        self._send({'type': 'reset'})

    def save(self, path: str = 'brain_save.pkl'):
        """Попросить сервер сохранить мозг в файл."""
        self._send({'type': 'save', 'path': path})

    def load(self, path: str):
        """Загрузить веса из файла (конфигурация должна совпадать)."""
        self._send({'type': 'load', 'path': path})

    def info(self) -> dict:
        """Получить информацию о состоянии мозга."""
        resp = self._send({'type': 'info'})
        return {k: v for k, v in resp.items() if k != 'type'}

    def close(self):
        if self._sock:
            self._sock.close()
            self._sock = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
