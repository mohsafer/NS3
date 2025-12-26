#!/usr/bin/env python3
"""
DQN Agent for NS3 OpenGym TCP Congestion Control
This script implements a Deep Q-Network (DQN) agent to learn optimal
TCP congestion control policies through reinforcement learning.
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import time

try:
    from ns3gym import ns3env
except ImportError:
    print("Error: ns3gym module not found. Please install ns-3 OpenGym module.")
    print("Installation: pip install ns3gym")
    sys.exit(1)

# Import training monitor (optional)
try:
    from training_monitor import TrainingMonitor
    MONITOR_AVAILABLE = True
except ImportError:
    print("Warning: training_monitor.py not found. Training monitoring disabled.")
    MONITOR_AVAILABLE = False


class DQNNetwork(nn.Module):
    """
    Deep Q-Network architecture for TCP congestion control
    """
    def __init__(self, state_size, action_size, hidden_size=128):
        super(DQNNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, action_size)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x


class ReplayBuffer:
    """
    Experience replay buffer for DQN training
    """
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones))
    
    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    """
    DQN Agent for learning TCP congestion control
    """
    def __init__(self, state_size, action_size, device='cpu'):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        
        # Hyperparameters
        self.gamma = 0.99           # Discount factor
        self.epsilon = 1.0          # Exploration rate
        self.epsilon_min = 0.01     # Minimum exploration rate
        self.epsilon_decay = 0.995  # Exploration decay rate
        self.learning_rate = 0.001  # Learning rate
        self.batch_size = 64        # Batch size for training
        self.target_update = 10     # Update target network every N episodes
        self.alpha = 0.1            # Entropy temperature (Soft Bellman)
        self.sigma = 0.05           # Brownian diffusion coefficient
        # Networks
        self.policy_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()
        
        # Replay buffer
        self.memory = ReplayBuffer(capacity=10000)
        
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def train(self):
        """
        Train the DQN network using experience replay
        """
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Current Q values
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        # with torch.no_grad():
        #     next_q = self.target_net(next_states).max(1)[0]
        #     target_q = rewards + (1 - dones) * self.gamma * next_q
        with torch.no_grad():
            next_q_values = self.target_net(next_states)
            
            # 1. Soft Value (Entropy-regularized value)
            # Log-Sum-Exp provides a smooth approximation of the maximum
            soft_value = self.alpha * torch.logsumexp(next_q_values / self.alpha, dim=1)
            
            # 2. Brownian Movement (Diffusion term)
            # Adds stochastic noise representing the "random walk" of network jitter
            brownian_increment = self.sigma * torch.randn_like(soft_value)
            
            # 3. Target Q calculation
            target_q = rewards + (1 - dones) * self.gamma * (soft_value + brownian_increment)



        # Compute loss
        loss = self.criterion(current_q.squeeze(), target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network with policy network weights
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """
        Decay exploration rate
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath):
        """
        Save model weights
        """
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """
        Load model weights
        """
        checkpoint = torch.load(filepath)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        print(f"Model loaded from {filepath}")


def train_dqn(env, agent, num_episodes=1000, max_steps=1000, monitor=None):
    """
    Training loop for DQN agent
    """
    episode_rewards = []
    episode_losses = []
    
    print("\n" + "="*60)
    print("Starting DQN Training")
    print("="*60)
    
    for episode in range(num_episodes):
        try:
            # Reset environment
            print(f"\nEpisode {episode + 1}/{num_episodes}: Resetting environment...")
            state = env.reset()
            
            # Convert state to numpy array if needed
            if not isinstance(state, np.ndarray):
                state = np.array(state)
            
            episode_reward = 0
            episode_loss = 0
            step_count = 0
            last_info = ""
            
            print(f"Initial state: {state}")
            
            for step in range(max_steps):
                # Select and perform action
                action = agent.select_action(state)
                
                # Take step in environment
                try:
                    next_state, reward, done, info = env.step(action)
                    
                    # Store last info for monitoring
                    if info:
                        last_info = str(info)
                    
                    # Convert next_state to numpy array if needed
                    if not isinstance(next_state, np.ndarray):
                        next_state = np.array(next_state)
                    
                    # Handle NaN or infinite values
                    if np.any(np.isnan(next_state)) or np.any(np.isinf(next_state)):
                        print(f"Warning: Invalid state detected at step {step}")
                        next_state = np.nan_to_num(next_state, 0.0)
                    
                    if np.isnan(reward) or np.isinf(reward):
                        print(f"Warning: Invalid reward detected at step {step}")
                        reward = 0.0
                    
                except Exception as e:
                    print(f"Error during step {step}: {e}")
                    break
                
                # Store transition in replay buffer
                agent.memory.push(state, action, reward, next_state, done)
                
                # Train the agent
                loss = agent.train()
                episode_loss += loss
                
                episode_reward += reward
                state = next_state
                step_count += 1
                
                # Print step info periodically
                if step % 50 == 0:
                    print(f"  Step {step}: Action={action}, Reward={reward:.3f}, Loss={loss:.4f}")
                
                if done:
                    print(f"Episode ended at step {step}")
                    break
            
            # Update target network periodically
            if episode % agent.target_update == 0:
                agent.update_target_network()
                print("Target network updated")
            
            # Decay exploration rate
            agent.decay_epsilon()
            
            # Record statistics
            episode_rewards.append(episode_reward)
            avg_loss = episode_loss / step_count if step_count > 0 else 0
            episode_losses.append(avg_loss)
            
            # Log to monitor if available
            if monitor:
                monitor.log_episode(episode, episode_reward, avg_loss, step_count, 
                                  agent.epsilon, last_info)
            
            # Print progress
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"\nEpisode {episode + 1}/{num_episodes} Summary:")
            print(f"  Total Reward: {episode_reward:.2f}")
            print(f"  Avg Reward (last 10): {avg_reward:.2f}")
            print(f"  Epsilon: {agent.epsilon:.3f}")
            print(f"  Avg Loss: {avg_loss:.4f}")
            print(f"  Steps: {step_count}")
            print(f"  Buffer size: {len(agent.memory)}")
            
            # Print statistics periodically
            if monitor and (episode + 1) % 100 == 0:
                monitor.print_statistics()
            
            # Save model periodically
            if (episode + 1) % 100 == 0:
                agent.save(f"dqn_tcp_episode_{episode + 1}.pth")
        
        except KeyboardInterrupt:
            print("\nTraining interrupted by user")
            break
        except Exception as e:
            print(f"\nError in episode {episode + 1}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    return episode_rewards, episode_losses


def test_dqn(env, agent, num_episodes=10):
    """
    Test trained DQN agent
    """
    test_rewards = []
    
    print("\n" + "="*60)
    print("Testing DQN Agent")
    print("="*60)
    
    for episode in range(num_episodes):
        try:
            state = env.reset()
            if not isinstance(state, np.ndarray):
                state = np.array(state)
                
            episode_reward = 0
            done = False
            step = 0
            
            while not done and step < 1000:
                # Select action without exploration
                action = agent.select_action(state, training=False)
                next_state, reward, done, info = env.step(action)
                
                if not isinstance(next_state, np.ndarray):
                    next_state = np.array(next_state)
                
                episode_reward += reward
                state = next_state
                step += 1
            
            test_rewards.append(episode_reward)
            print(f"Test Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
        
        except Exception as e:
            print(f"Error in test episode {episode + 1}: {e}")
    
    if test_rewards:
        avg_reward = np.mean(test_rewards)
        print(f"\nAverage Test Reward: {avg_reward:.2f}")
    
    return test_rewards


def main():
    parser = argparse.ArgumentParser(description='DQN Agent for NS3 OpenGym TCP Control')
    parser.add_argument('--port', type=int, default=5555, help='OpenGym port (default: 5555)')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of training episodes')
    parser.add_argument('--test', action='store_true', help='Test mode (load existing model)')
    parser.add_argument('--load', type=str, default=None, help='Load model from file')
    parser.add_argument('--save', type=str, default='dqn_tcp_final.pth', help='Save model to file')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], 
                        help='Device to use for training')
    parser.add_argument('--max-steps', type=int, default=1000, help='Max steps per episode')
    parser.add_argument('--log-dir', type=str, default='./logs', help='Directory for logs and plots')
    parser.add_argument('--no-monitor', action='store_true', help='Disable training monitoring')
    args = parser.parse_args()
    
    # Check for CUDA availability
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU instead")
        device = 'cpu'
    
    print(f"Using device: {device}")
    
    # Initialize environment
    print(f"\nConnecting to NS3 OpenGym on port {args.port}...")
    print("Make sure NS3 simulation is running first!")
    print("Run: ./ns3 run \"tcp --openGym=1 --simTime=50\"\n")
    
    try:
        env = ns3env.Ns3Env(port=args.port, startSim=False, simSeed=0, 
                           simArgs={"--openGym": "1"}, debug=False)
        print("Successfully connected to NS3!")
    except Exception as e:
        print(f"Error connecting to NS3: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure NS3 simulation is running")
        print("2. Check if port 5555 is available")
        print("3. Verify ns3gym is installed correctly")
        sys.exit(1)
    
    # Get state and action space dimensions
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    print(f"\nEnvironment Information:")
    print(f"  State size: {state_size}")
    print(f"  Action size: {action_size}")
    print(f"  Observation space: {env.observation_space}")
    print(f"  Action space: {env.action_space}")
    
    # Initialize agent
    agent = DQNAgent(state_size, action_size, device=device)
    
    # Load existing model if specified
    if args.load:
        if os.path.exists(args.load):
            agent.load(args.load)
        else:
            print(f"Warning: Model file {args.load} not found, starting with random weights")
    
    # Test or train mode
    if args.test:
        print("\nRunning in test mode...")
        if not args.load:
            print("Warning: Testing with random weights (no model loaded)")
        test_dqn(env, agent, num_episodes=10)
    else:
        # Initialize training monitor
        monitor = None
        if MONITOR_AVAILABLE and not args.no_monitor:
            monitor = TrainingMonitor(log_dir=args.log_dir, plot_interval=50)
            print(f"Training monitor enabled. Logs will be saved to: {args.log_dir}")
        elif args.no_monitor:
            print("Training monitor disabled by user")
        
        print("\nStarting training...")
        print(f"Episodes: {args.episodes}")
        print(f"Max steps per episode: {args.max_steps}")
        
        start_time = time.time()
        episode_rewards, episode_losses = train_dqn(env, agent, 
                                                     num_episodes=args.episodes,
                                                     max_steps=args.max_steps,
                                                     monitor=monitor)
        elapsed_time = time.time() - start_time
        
        # Save final model
        agent.save(args.save)
        
        # Final statistics and plots
        if monitor:
            print("\nGenerating final plots and statistics...")
            monitor.print_statistics()
            monitor.plot_training_progress(save_only=False)
            monitor.save_summary()
        
        # Print summary statistics
        print("\n" + "="*60)
        print("Training Summary")
        print("="*60)
        print(f"Total Episodes: {args.episodes}")
        print(f"Total Time: {elapsed_time/60:.2f} minutes")
        if episode_rewards:
            print(f"Average Reward (all): {np.mean(episode_rewards):.2f}")
            print(f"Average Reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
            print(f"Best Episode Reward: {np.max(episode_rewards):.2f}")
        print(f"Final Epsilon: {agent.epsilon:.3f}")
        print(f"Final Buffer Size: {len(agent.memory)}")
    
    try:
        env.close()
    except:
        pass
    
    print("\nDone!")


if __name__ == "__main__":
    main()