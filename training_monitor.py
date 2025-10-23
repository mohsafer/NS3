#!/usr/bin/env python3
"""
Training visualization and monitoring tool for DQN agent
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime


class TrainingMonitor:
    """Monitor and visualize DQN training progress"""
    
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_lengths = []
        self.epsilon_values = []
        
    def log_episode(self, episode, reward, loss, length, epsilon):
        """Log data from a single episode"""
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)
        
        # Save to file
        log_data = {
            'episode': episode,
            'reward': reward,
            'loss': loss,
            'length': length,
            'epsilon': epsilon,
            'timestamp': datetime.now().isoformat()
        }
        
        log_file = os.path.join(self.log_dir, 'training_log.jsonl')
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
    
    def plot_training_progress(self, window_size=100):
        """Plot training progress with moving averages"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = np.arange(len(self.episode_rewards))
        
        # Plot rewards
        axes[0, 0].plot(episodes, self.episode_rewards, alpha=0.3, label='Raw')
        if len(self.episode_rewards) >= window_size:
            moving_avg = self._moving_average(self.episode_rewards, window_size)
            axes[0, 0].plot(episodes[window_size-1:], moving_avg, label=f'MA-{window_size}')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot losses
        axes[0, 1].plot(episodes, self.episode_losses, alpha=0.3, label='Raw')
        if len(self.episode_losses) >= window_size:
            moving_avg = self._moving_average(self.episode_losses, window_size)
            axes[0, 1].plot(episodes[window_size-1:], moving_avg, label=f'MA-{window_size}')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot epsilon
        axes[1, 0].plot(episodes, self.epsilon_values)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].grid(True)
        
        # Plot episode lengths
        axes[1, 1].plot(episodes, self.episode_lengths, alpha=0.3, label='Raw')
        if len(self.episode_lengths) >= window_size:
            moving_avg = self._moving_average(self.episode_lengths, window_size)
            axes[1, 1].plot(episodes[window_size-1:], moving_avg, label=f'MA-{window_size}')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Steps')
        axes[1, 1].set_title('Episode Length')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(os.path.join(self.log_dir, f'training_progress_{timestamp}.png'))
        plt.show()
    
    def _moving_average(self, data, window_size):
        """Calculate moving average"""
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def print_statistics(self, recent_episodes=100):
        """Print training statistics"""
        if len(self.episode_rewards) == 0:
            print("No training data available")
            return
        
        print("\n" + "="*60)
        print("Training Statistics")
        print("="*60)
        
        print(f"\nTotal Episodes: {len(self.episode_rewards)}")
        
        # Overall statistics
        print("\nOverall Performance:")
        print(f"  Average Reward: {np.mean(self.episode_rewards):.2f}")
        print(f"  Max Reward: {np.max(self.episode_rewards):.2f}")
        print(f"  Min Reward: {np.min(self.episode_rewards):.2f}")
        print(f"  Std Reward: {np.std(self.episode_rewards):.2f}")
        
        # Recent performance
        if len(self.episode_rewards) >= recent_episodes:
            recent_rewards = self.episode_rewards[-recent_episodes:]
            print(f"\nRecent Performance (last {recent_episodes} episodes):")
            print(f"  Average Reward: {np.mean(recent_rewards):.2f}")
            print(f"  Max Reward: {np.max(recent_rewards):.2f}")
            print(f"  Min Reward: {np.min(recent_rewards):.2f}")
            print(f"  Std Reward: {np.std(recent_rewards):.2f}")
        
        # Learning progress
        if len(self.episode_rewards) >= 200:
            first_100 = np.mean(self.episode_rewards[:100])
            last_100 = np.mean(self.episode_rewards[-100:])
            improvement = ((last_100 - first_100) / abs(first_100)) * 100
            print(f"\nLearning Progress:")
            print(f"  First 100 episodes avg: {first_100:.2f}")
            print(f"  Last 100 episodes avg: {last_100:.2f}")
            print(f"  Improvement: {improvement:+.1f}%")
        
        # Current state
        if len(self.episode_rewards) > 0:
            print(f"\nCurrent State:")
            print(f"  Latest Reward: {self.episode_rewards[-1]:.2f}")
            print(f"  Latest Loss: {self.episode_losses[-1]:.4f}")
            print(f"  Current Epsilon: {self.epsilon_values[-1]:.3f}")
        
        print("="*60 + "\n")


def load_training_log(log_file='./logs/training_log.jsonl'):
    """Load training log from file and create visualizations"""
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return
    
    monitor = TrainingMonitor()
    
    with open(log_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            monitor.episode_rewards.append(data['reward'])
            monitor.episode_losses.append(data['loss'])
            monitor.episode_lengths.append(data['length'])
            monitor.epsilon_values.append(data['epsilon'])
    
    print(f"Loaded {len(monitor.episode_rewards)} episodes from log")
    monitor.print_statistics()
    monitor.plot_training_progress()


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Training monitor and visualization')
    parser.add_argument('--log-file', type=str, default='./logs/training_log.jsonl',
                       help='Path to training log file')
    args = parser.parse_args()
    
    load_training_log(args.log_file)
