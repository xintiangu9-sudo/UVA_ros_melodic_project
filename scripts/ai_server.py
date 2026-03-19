import socket
import json
import random

HOST = "127.0.0.1"
PORT = 9999

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((HOST, PORT))
server.listen(1)

print("AI Server Started... Waiting for ROS connection")

conn, addr = server.accept()
print("Connected:", addr)

buffer = ""

while True:
    try:
        data = conn.recv(4096).decode()
        if not data:
            continue

        buffer += data

        if "\n" not in buffer:
            continue

        line, buffer = buffer.split("\n", 1)
        obs = json.loads(line)

        scan = obs["scan"]
        state = obs["state"]

        # 简单避障策略
        front = min(scan)

        if front < 1.0:
            action = {
                "linear": 0.0,
                "angular": random.uniform(-1.0, 1.0)
            }
        else:
            action = {
                "linear": 1.0,
                "angular": 0.0
            }

        conn.sendall((json.dumps(action) + "\n").encode())

    except Exception as e:
        print("Error:", e)
        break

conn.close()
