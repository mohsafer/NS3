#!/usr/bin/env python3
"""
DQN Agent for NS-3 OpenGym TCP Congestion Control
OPTION 1 IMPLEMENTATION:
  - One ns-3 process per episode
  - Python controls ns-3 lifecycle
"""

import os
import sys
import time
import argparse
import subprocess
import random
import numpy as np
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from ns3gym import ns3env


# =========================
# NS-3 LAUNCHER
# =========================
def launch_ns3(port, sim_time):
    """
    Launch ns-3 simulation as a subprocess
    """
    cmd = [
        "/users/mosafer/ns-allinone-3.40/ns-3.40/ns3", "run",
        f"tcp --openGym=1 --openGymPort={port} --simTime={sim_time}"
    ]
    print("Launching ns-3:", " ".join(cmd))
    return subprocess.Popen(cmd)


# =========================
# DQN NETWORK
# =========================
class DQNNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
        )

    def forward(self, x):
        return self.net(x)


# =========================
# REPLAY BUFFER
# =========================
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, d):
        self.buffer.append((s, a, r, ns, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            np.array(s),
            np.array(a),
            np.array(r),
            np.array(ns),
            np.array(d),
        )

    def __len__(self):
        return len(self.buffer)


# =========================
# DQN AGENT
# =========================
class DQNAgent:
    def __init__(self, state_size, action_size, device="cpu"):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device

        # Hyperparameters
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.lr = 1e-3
        self.batch_size = 64
        self.target_update = 10

        self.policy_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()
        self.memory = ReplayBuffer()

    def select_action(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            return self.policy_net(s).argmax().item()

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0

        s, a, r, ns, d = self.memory.sample(self.batch_size)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.LongTensor(a).to(self.device)
        r = torch.FloatTensor(r).to(self.device)
        ns = torch.FloatTensor(ns).to(self.device)
        d = torch.FloatTensor(d).to(self.device)

        q = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze()

        with torch.no_grad():
            q_next = self.target_net(ns).max(1)[0]
            q_target = r + (1 - d) * self.gamma * q_next

        loss = self.loss_fn(q, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon
        }, path)
        print(f"Model saved: {path}")


# =========================
# TRAINING LOOP (OPTION 1)
# =========================
def train(agent, episodes, max_steps, port, sim_time):
    rewards = []

    for ep in range(episodes):
        print(f"\n========== Episode {ep+1}/{episodes} ==========")

        # 1) Launch ns-3
        ns3_proc = launch_ns3(port, sim_time)

        # 2) Connect OpenGym
        env = ns3env.Ns3Env(
            port=port,
            startSim=False,
            simSeed=ep,
            simArgs={"--openGym": "1"},
            debug=False,
        )

        state = np.array(env.reset())
        total_reward = 0.0
        total_loss = 0.0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            next_state = np.array(next_state)

            agent.memory.push(state, action, reward, next_state, done)
            loss = agent.train_step()

            state = next_state
            total_reward += reward
            total_loss += loss

            if done:
                break

        # 3) Cleanup
        env.close()
        ns3_proc.wait()

        # 4) Updates
        if ep % agent.target_update == 0:
            agent.update_target()

        agent.decay_epsilon()
        rewards.append(total_reward)

        avg_loss = total_loss / max(1, step + 1)
        print(f"Reward: {total_reward:.2f}")
        print(f"Avg loss: {avg_loss:.4f}")
        print(f"Epsilon: {agent.epsilon:.3f}")
        print(f"Replay buffer: {len(agent.memory)}")

    return rewards


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--sim-time", type=int, default=1500)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--save", default="dqn_tcp_final.pth")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("Using device:", device)

    # Temporary env to read spaces
    print("Starting temporary ns-3 to read spaces...")
    tmp_proc = launch_ns3(args.port, args.sim_time)
    tmp_env = ns3env.Ns3Env(port=args.port, startSim=False)
    state_size = tmp_env.observation_space.shape[0]
    action_size = tmp_env.action_space.n
    tmp_env.close()
    tmp_proc.wait()

    print("State size:", state_size)
    print("Action size:", action_size)

    agent = DQNAgent(state_size, action_size, device=device)

    start = time.time()
    rewards = train(
        agent,
        episodes=args.episodes,
        max_steps=args.max_steps,
        port=args.port,
        sim_time=args.sim_time,
    )
    end = time.time()

    agent.save(args.save)

    print("\nTraining finished")
    print(f"Total time: {(end-start)/60:.2f} minutes")
    print(f"Average reward: {np.mean(rewards):.2f}")


if __name__ == "__main__":
    main()
