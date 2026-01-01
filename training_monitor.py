#!/usr/bin/env python3
"""
Training visualization and monitoring tool for DQN agent
"""
#NEW Modified File date: 2024-06-10 12:00:00.000000000 +0000
#TODO: Implement advanced visualization features + Science plots
import matplotlib.pyplot as plt
import numpy as np
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots


class TrainingMonitor:
    """Monitor and visualize DQN training progress"""
    
    def __init__(self, log_dir='./logs', plot_interval=50):
        self.log_dir = log_dir
        self.plot_interval = plot_interval
        os.makedirs(log_dir, exist_ok=True)
        
        self.episode_rewards = []
        self.episode_losses = []
        self.episode_lengths = []
        self.epsilon_values = []
        self.throughput_values = []
        self.rtt_values = []
        self.cwnd_values = []
        
        # Initialize log file
        self.log_file = os.path.join(self.log_dir, 'training_log.jsonl')
        
    def log_episode(self, episode, reward, loss, length, epsilon, extra_info=None):
        """Log data from a single episode"""
        self.episode_rewards.append(reward)
        self.episode_losses.append(loss)
        self.episode_lengths.append(length)
        self.epsilon_values.append(epsilon)
        
        # Extract network metrics if available
        if extra_info:
            throughput = self._extract_metric(extra_info, 'throughput')
            rtt = self._extract_metric(extra_info, 'rtt')
            cwnd = self._extract_metric(extra_info, 'cwnd')
            
            self.throughput_values.append(throughput)
            self.rtt_values.append(rtt)
            self.cwnd_values.append(cwnd)
        
        # Save to file
        log_data = {
            'episode': episode,
            'reward': float(reward),
            'loss': float(loss),
            'length': int(length),
            'epsilon': float(epsilon),
            'timestamp': datetime.now().isoformat()
        }
        
        if extra_info:
            log_data['extra_info'] = str(extra_info)
        
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(log_data) + '\n')
        
        # Plot periodically
        if (episode + 1) % self.plot_interval == 0:
            self.plot_training_progress(save_only=True)
    
    def _extract_metric(self, info_str, metric_name):
        """Extract a metric value from the info string"""
        try:
            if metric_name in info_str:
                # Parse format: "metric_name=value"
                parts = info_str.split(f'{metric_name}=')
                if len(parts) > 1:
                    value_str = parts[1].split(',')[0].split()[0]
                    return float(value_str)
        except:
            pass
        return 0.0
    def smooth(self, y, box_pts=6):
        """Apply moving average smoothing"""
        y = np.array(y)  # Ensure it's a numpy array
        
        # If data is too short, return as-is
        if len(y) < box_pts:
            return y
        
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def plot_training_progress(self, window_size=50, save_only=False):
        """Plot training progress with moving averages"""
        if len(self.episode_rewards) < 2:
            return
        
        #fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig, axes = plt.subplots(3, 2, figsize=(16, 16))

        episodes = np.arange(len(self.episode_rewards))
        
        # Plot rewards
        axes[0, 0].plot(episodes, self.smooth(self.episode_rewards), label='Raw', linewidth=2, color='orange')
        # if len(self.episode_rewards) >= window_size:
        #     moving_avg = self._moving_average(self.episode_rewards, window_size)
        #     axes[0, 0].plot(episodes[window_size-1:], moving_avg, label=f'MA-{window_size}', linewidth=2)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot losses
        #valid_losses = [l for l in self.episode_losses if l > 0]
        valid_losses = [max(1e-6, l)for l in self.episode_losses if l > 0]

        if valid_losses:
            axes[0, 1].plot(episodes[:len(valid_losses)], valid_losses, label='Raw', linewidth=2, color='orange')
            if len(valid_losses) >= window_size:
                moving_avg = self._moving_average(valid_losses, window_size)
                axes[0, 1].plot(episodes[window_size-1:len(valid_losses)], moving_avg, 
                              label=f'MA-{window_size}', linewidth=2)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Loss')
        axes[0, 1].set_title('Training Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_yscale('linear')
        
        # Plot epsilon
        axes[1, 0].plot(episodes, self.epsilon_values, linewidth=2, color='orange')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim([0, 1.1])
        
        # # Plot episode lengths
        # axes[1, 1].plot(episodes, self.episode_lengths, label='Raw', linewidth=2, color='orange')
        # if len(self.episode_lengths) >= window_size:
        #     moving_avg = self._moving_average(self.episode_lengths, window_size)
        #     axes[1, 1].plot(episodes[window_size-1:], moving_avg, label=f'MA-{window_size}', linewidth=2, color='blue')
        # axes[1, 1].set_xlabel('Episode')
        # axes[1, 1].set_ylabel('Steps')
        # axes[1, 1].set_title('Episode Length')
        # axes[1, 1].legend()
        # axes[1, 1].grid(True, alpha=0.3)
        
        # Plot throughput if available
        if self.throughput_values and any(self.throughput_values):
            axes[1, 1].plot(episodes[:len(self.throughput_values)], self.smooth(self.throughput_values), 
                                label='Raw', linewidth=2, color='orange')
            if len(self.throughput_values) >= window_size:
                moving_avg = self._moving_average(self.throughput_values, window_size)
                axes[1, 1].plot(episodes[window_size-1:len(self.throughput_values)], moving_avg, 
                              label=f'MA-{window_size}', linewidth=2)
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Throughput (Mbps)')
            axes[1, 1].set_title('Network Throughput')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No throughput data', 
                          ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Network Throughput (No Data)')
        # Plot RTT if available
        if self.rtt_values and any(self.rtt_values):
            axes[2, 0].plot(
                episodes[:len(self.rtt_values)],
                self.smooth(self.rtt_values),
                label='Raw',
                linewidth=2, color='blue'
            )
            if len(self.rtt_values) >= window_size:
                moving_avg = self._moving_average(self.rtt_values, window_size)
                axes[2, 0].plot(
                    episodes[window_size-1:len(self.rtt_values)],
                    moving_avg,
                    label=f'MA-{window_size}',
                    linewidth=2, color='green'
                )
            axes[2, 0].set_xlabel('Episode')
            axes[2, 0].set_ylabel('RTT (s)')
            axes[2, 0].set_title('Round-Trip Time (RTT)')
            axes[2, 0].legend()
            axes[2, 0].grid(True, alpha=0.3)
        else:
            axes[2, 0].text(0.5, 0.5, 'No RTT data',
                    ha='center', va='center', transform=axes[2, 0].transAxes)


        

        
        # Plot CWND if available
        if self.cwnd_values and any(self.cwnd_values):
            axes[2, 1].plot(
                episodes[:len(self.cwnd_values)],
                self.cwnd_values,
                label='Raw',
                linewidth=2, color='green'
            )
            if len(self.cwnd_values) >= window_size:
                moving_avg = self._moving_average(self.cwnd_values, window_size)
                axes[2, 1].plot(
                    episodes[window_size-1:len(self.cwnd_values)],
                    moving_avg,
                    label=f'MA-{window_size}',
                    linewidth=2
                )
            axes[2, 1].set_xlabel('Episode')
            axes[2, 1].set_ylabel('CWND')
            axes[2, 1].set_title('Congestion Window (CWND)')
            axes[2, 1].legend()
            axes[2, 1].grid(True, alpha=0.3)
        else:
            axes[2, 1].text(0.5, 0.5, 'No CWND data',
                            ha='center', va='center', transform=axes[2, 1].transAxes)

        plt.tight_layout()
        # Save figure
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_path = os.path.join(self.log_dir, f'training_progress_{timestamp}.png')
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        
        # Also save a "latest" version
        latest_path = os.path.join(self.log_dir, 'training_progress_latest.png')
        plt.savefig(latest_path, dpi=100, bbox_inches='tight')
        
        if not save_only:
            plt.show()
        
        plt.close(fig)
        
        print(f"Training plot saved to: {latest_path}")
    
    def _moving_average(self, data, window_size):
        """Calculate moving average"""
        if len(data) < window_size:
            return data
        return np.convolve(data, np.ones(window_size)/window_size, mode='valid')
    
    def print_statistics(self, recent_episodes=100):
        """Print training statistics"""
        if len(self.episode_rewards) == 0:
            print("No training data available")
            return
        
        print("\n" + "="*70)
        print("Training Statistics")
        print("="*70)
        
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
            
            # Calculate trend
            if len(self.episode_rewards) >= recent_episodes * 2:
                prev_avg = np.mean(self.episode_rewards[-recent_episodes*2:-recent_episodes])
                curr_avg = np.mean(recent_rewards)
                trend = ((curr_avg - prev_avg) / abs(prev_avg)) * 100 if prev_avg != 0 else 0
                print(f"  Trend vs previous {recent_episodes}: {trend:+.1f}%")
        
        # Learning progress
        if len(self.episode_rewards) >= 200:
            first_100 = np.mean(self.episode_rewards[:100])
            last_100 = np.mean(self.episode_rewards[-100:])
            improvement = ((last_100 - first_100) / abs(first_100)) * 100 if first_100 != 0 else 0
            print(f"\nLearning Progress:")
            print(f"  First 100 episodes avg: {first_100:.2f}")
            print(f"  Last 100 episodes avg: {last_100:.2f}")
            print(f"  Improvement: {improvement:+.1f}%")
        
        # Network metrics if available
        if self.throughput_values and any(self.throughput_values):
            print(f"\nNetwork Metrics:")
            print(f"  Average Throughput: {np.mean([t for t in self.throughput_values if t > 0]):.2f} Mbps")
            if self.rtt_values:
                print(f"  Average RTT: {np.mean([r for r in self.rtt_values if r > 0]):.4f} s")
            if self.cwnd_values:
                print(f"  Average CWND: {np.mean([c for c in self.cwnd_values if c > 0]):.0f}")
        
        # Current state
        if len(self.episode_rewards) > 0:
            print(f"\nCurrent State:")
            print(f"  Latest Reward: {self.episode_rewards[-1]:.2f}")
            if self.episode_losses[-1] > 0:
                print(f"  Latest Loss: {self.episode_losses[-1]:.4f}")
            print(f"  Current Epsilon: {self.epsilon_values[-1]:.3f}")
            print(f"  Buffer Size: Available in agent")
        
        print("="*70 + "\n")
    
    def save_summary(self):
        """Save training summary to JSON file"""
        if len(self.episode_rewards) == 0:
            return
        
        summary = {
            'total_episodes': len(self.episode_rewards),
            'final_epsilon': float(self.epsilon_values[-1]),
            'rewards': {
                'mean': float(np.mean(self.episode_rewards)),
                'std': float(np.std(self.episode_rewards)),
                'min': float(np.min(self.episode_rewards)),
                'max': float(np.max(self.episode_rewards)),
            },
            'last_100_reward_mean': float(np.mean(self.episode_rewards[-100:])) if len(self.episode_rewards) >= 100 else None,
            'timestamp': datetime.now().isoformat()
        }
        
        summary_file = os.path.join(self.log_dir, 'training_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_file}")


def load_training_log(log_file='./logs/training_log.jsonl'):
    """Load training log from file and create visualizations"""
    
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return None
    
    log_dir = os.path.dirname(log_file)
    monitor = TrainingMonitor(log_dir=log_dir if log_dir else './logs')
    
    print(f"Loading training log from: {log_file}")
    
    with open(log_file, 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                monitor.episode_rewards.append(data['reward'])
                monitor.episode_losses.append(data['loss'])
                monitor.episode_lengths.append(data['length'])
                monitor.epsilon_values.append(data['epsilon'])
                
                # Extract network metrics if available
                if 'extra_info' in data:
                    info = data['extra_info']
                    monitor.throughput_values.append(monitor._extract_metric(info, 'throughput'))
                    monitor.rtt_values.append(monitor._extract_metric(info, 'rtt'))
                    monitor.cwnd_values.append(monitor._extract_metric(info, 'cwnd'))
            except Exception as e:
                print(f"Warning: Error parsing log line: {e}")
                continue
    
    print(f"Loaded {len(monitor.episode_rewards)} episodes from log")
    monitor.print_statistics()
    monitor.plot_training_progress(save_only=False)
    
    return monitor


def compare_training_runs(log_files, labels=None):
    """Compare multiple training runs"""
    
    if labels is None:
        labels = [f"Run {i+1}" for i in range(len(log_files))]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for log_file, label in zip(log_files, labels):
        if not os.path.exists(log_file):
            print(f"Warning: {log_file} not found, skipping")
            continue
        
        rewards = []
        losses = []
        
        with open(log_file, 'r') as f:
            for line in f:
                try:
                    data = json.loads(line)
                    rewards.append(data['reward'])
                    losses.append(data['loss'])
                except:
                    continue
        
        episodes = np.arange(len(rewards))
        
        # Plot rewards
        if len(rewards) >= 50:
            ma_rewards = np.convolve(rewards, np.ones(50)/50, mode='valid')
            axes[0, 0].plot(episodes[49:], ma_rewards, label=label, linewidth=2)
        
        # Plot losses
        valid_losses = [l for l in losses if l > 0]
        if len(valid_losses) >= 50:
            ma_losses = np.convolve(valid_losses, np.ones(50)/50, mode='valid')
            axes[0, 1].plot(episodes[49:len(valid_losses)], ma_losses, label=label, linewidth=2)
        
        # Plot cumulative reward
        cumulative = np.cumsum(rewards)
        axes[1, 0].plot(episodes, cumulative, label=label, linewidth=2)
        
        # Statistics
        print(f"\n{label} Statistics:")
        print(f"  Total Episodes: {len(rewards)}")
        print(f"  Mean Reward: {np.mean(rewards):.2f}")
        print(f"  Final 100 Mean: {np.mean(rewards[-100:]):.2f}")
    
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward (MA-50)')
    axes[0, 0].set_title('Reward Comparison')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Loss (MA-50)')
    axes[0, 1].set_title('Loss Comparison')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Cumulative Reward')
    axes[1, 0].set_title('Cumulative Reward')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Hide unused subplot
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig('./logs/training_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\nComparison plot saved to: ./logs/training_comparison.png")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Training monitor and visualization')
    parser.add_argument('--log-file', type=str, default='./logs/training_log.jsonl',
                       help='Path to training log file')
    parser.add_argument('--compare', type=str, nargs='+',
                       help='Compare multiple training runs')
    parser.add_argument('--labels', type=str, nargs='+',
                       help='Labels for comparison runs')
    args = parser.parse_args()
    
    if args.compare:
        compare_training_runs(args.compare, args.labels)
    else:
        monitor = load_training_log(args.log_file)
        if monitor:
            monitor.save_summary()