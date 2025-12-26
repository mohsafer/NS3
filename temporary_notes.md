cd ~/ns-allinone-3.40/ns-3.40/scratch/deepRL

 
# Fix the class name
sed -i 's/TcpWestwood/TcpWestwoodPlus/g' tcp.cc

# Fix the include
sed -i 's/#include "ns3\/tcp-westwood.h"/#include "ns3\/tcp-westwood-plus.h"/g' tcp.cc

# Comment out the ProtocolType line (it doesn't exist in TcpWestwoodPlus)
sed -i 's/Config::SetDefault ("ns3::TcpWestwood::ProtocolType"/\/\/ Config::SetDefault ("ns3::TcpWestwoodPlus::ProtocolType"/g' tcp.cc

sed -i 's/.*TcpWestwoodPlus::ProtocolType.*/\/\/ &/' tcp.cc



./ns3 run "tcp --openGym=1 --simTime=150"

./ns3 run "tcp --openGym=1 --simTime=50"


 python3 dqn_agent.py --episodes 2000


##################################################################
##################################################################



# DQN Training Monitor - Usage Guide

## Overview
The training monitor provides real-time visualization and logging of your DQN agent's training progress, including rewards, losses, network metrics, and exploration rates.

## Files
- `dqn_agent.py` - Main DQN training script (updated with monitoring)
- `training_monitor.py` - Monitoring and visualization tool
- `tcp.cc` - NS-3 simulation with OpenGym integration

## Installation

### Required Packages
```bash
pip install matplotlib numpy torch ns3gym
```

## Basic Usage

### 1. Training with Monitoring (Default)

**Terminal 1 - Start NS-3 Simulation:**
```bash
./ns3 run "tcp --openGym=1 --simTime=50"
```

**Terminal 2 - Start DQN Training:**
```bash
python3 dqn_agent.py --episodes 1000 --log-dir ./logs
```

This will:
- Train for 1000 episodes
- Save logs to `./logs/training_log.jsonl`
- Generate plots every 50 episodes
- Save plots to `./logs/training_progress_latest.png`
- Display final statistics and plots

### 2. Training WITHOUT Monitoring

```bash
python3 dqn_agent.py --episodes 1000 --no-monitor
```

### 3. Custom Log Directory

```bash
python3 dqn_agent.py --episodes 1000 --log-dir ./my_experiment_1
```

## Monitoring Features

### Real-time Metrics Tracked
1. **Episode Rewards** - Total reward per episode with moving average
2. **Training Loss** - Average loss per episode
3. **Exploration Rate (Epsilon)** - Decay over time
4. **Episode Length** - Number of steps per episode
5. **Network Throughput** - From NS-3 simulation (if available)
6. **RTT** - Round-trip time from simulation
7. **CWND** - Congestion window size

### Output Files

#### Training Log (`training_log.jsonl`)
JSON Lines format with one episode per line:
```json
{"episode": 0, "reward": 12.5, "loss": 0.023, "length": 450, "epsilon": 0.995, "timestamp": "..."}
{"episode": 1, "reward": 15.3, "loss": 0.021, "length": 480, "epsilon": 0.990, "timestamp": "..."}
```

#### Training Plots
- `training_progress_latest.png` - Most recent training progress
- `training_progress_YYYYMMDD_HHMMSS.png` - Timestamped snapshots

#### Training Summary (`training_summary.json`)
Final statistics in JSON format

## Advanced Usage

### 1. View Training Progress (During or After Training)

```bash
python3 training_monitor.py --log-file ./logs/training_log.jsonl
```

This will:
- Load the training log
- Print comprehensive statistics
- Display interactive plots
- Save visualization

### 2. Compare Multiple Training Runs

```bash
python3 training_monitor.py --compare \
    ./experiment1/training_log.jsonl \
    ./experiment2/training_log.jsonl \
    ./experiment3/training_log.jsonl \
    --labels "Baseline" "High LR" "Low Epsilon"
```

Output: Comparison plots showing all runs side-by-side

### 3. Custom Training Parameters

```bash
python3 dqn_agent.py \
    --episodes 2000 \
    --max-steps 1000 \
    --log-dir ./experiments/run_001 \
    --save dqn_model_v1.pth \
    --port 5555
```

### 4. Resume Training from Checkpoint

```bash
python3 dqn_agent.py \
    --episodes 1000 \
    --load dqn_tcp_episode_500.pth \
    --save dqn_tcp_continued.pth \
    --log-dir ./logs_continued
```

### 5. Testing Mode (No Training)

```bash
python3 dqn_agent.py \
    --test \
    --load dqn_tcp_final.pth \
    --episodes 10
```

## Generated Plots

### Main Training Plot (6 subplots)

1. **Episode Rewards** (top-left)
   - Raw rewards (light line)
   - 50-episode moving average (bold line)
   - Shows learning progress

2. **Training Loss** (top-right)
   - Log scale for better visibility
   - Moving average overlay
   - Indicates convergence

3. **Exploration Rate** (middle-left)
   - Epsilon decay over time
   - Should decrease from 1.0 to ~0.01

4. **Episode Length** (middle-right)
   - Steps per episode
   - Longer = agent survives longer

5. **Network Throughput** (bottom-left)
   - Only if NS-3 metrics available
   - Shows network performance

6. **Reward Distribution** (bottom-right)
   - Histogram of all rewards
   - Mean line overlay

## Statistics Output

```
======================================================================
Training Statistics
======================================================================

Total Episodes: 1000

Overall Performance:
  Average Reward: 45.23
  Max Reward: 89.45
  Min Reward: -12.34
  Std Reward: 15.67

Recent Performance (last 100 episodes):
  Average Reward: 67.89
  Max Reward: 89.45
  Min Reward: 52.10
  Std Reward: 8.34
  Trend vs previous 100: +24.5%

Learning Progress:
  First 100 episodes avg: 23.45
  Last 100 episodes avg: 67.89
  Improvement: +189.5%

Network Metrics:
  Average Throughput: 3.45 Mbps
  Average RTT: 0.0042 s
  Average CWND: 156

Current State:
  Latest Reward: 78.90
  Latest Loss: 0.0234
  Current Epsilon: 0.012
======================================================================
```

## Tips for Effective Monitoring

### 1. Check Plots Every 50 Episodes
Monitor `./logs/training_progress_latest.png` to catch issues early

### 2. Watch for These Patterns

**Good Signs:**
- ✅ Rewards trending upward
- ✅ Loss decreasing and stabilizing
- ✅ Consistent episode lengths
- ✅ Network throughput improving

**Warning Signs:**
- ⚠️ Rewards oscillating wildly → Reduce learning rate
- ⚠️ Loss not decreasing → Check reward function
- ⚠️ Episode lengths = 1 → Environment ending too early
- ⚠️ All zeros in observations → Check NS-3 traces

### 3. Adjust Hyperparameters Based on Plots

If rewards plateau:
- Decrease epsilon decay rate (explore more)
- Increase learning rate slightly
- Increase network capacity (hidden layers)

If training unstable:
- Decrease learning rate
- Increase batch size
- Add gradient clipping (already implemented)

### 4. Save Multiple Checkpoints

```bash
# Training saves automatically every 100 episodes
# Files: dqn_tcp_episode_100.pth, dqn_tcp_episode_200.pth, etc.
```

## Troubleshooting

### "training_monitor.py not found"
```bash
# Make sure both files are in the same directory
ls -la dqn_agent.py training_monitor.py
```

### No plots appearing
```bash
# Check if matplotlib backend works
python3 -c "import matplotlib.pyplot as plt; plt.plot([1,2]); plt.savefig('test.png')"
```

### Log file not created
```bash
# Check directory permissions
ls -la ./logs/
# Should show training_log.jsonl
```

### Plots show no data
```bash
# Verify training is logging properly
tail -f ./logs/training_log.jsonl
# Should show new lines as training progresses
```

## Example Workflow

### Full Experiment Run

```bash
# 1. Start NS-3
cd /path/to/ns-3
./ns3 run "tcp --openGym=1 --simTime=50"

# 2. In another terminal, start training
cd /path/to/dqn
python3 dqn_agent.py \
    --episodes 1000 \
    --max-steps 500 \
    --log-dir ./experiments/baseline_run \
    --save models/baseline_final.pth

# 3. Monitor progress during training
# Check: ./experiments/baseline_run/training_progress_latest.png

# 4. After training, analyze results
python3 training_monitor.py \
    --log-file ./experiments/baseline_run/training_log.jsonl

# 5. Test the trained model
python3 dqn_agent.py \
    --test \
    --load models/baseline_final.pth \
    --episodes 20
```

### Compare Experiments

```bash
# Run multiple experiments with different settings
python3 dqn_agent.py --episodes 500 --log-dir exp1 ...
python3 dqn_agent.py --episodes 500 --log-dir exp2 ...
python3 dqn_agent.py --episodes 500 --log-dir exp3 ...

# Compare all runs
python3 training_monitor.py --compare \
    exp1/training_log.jsonl \
    exp2/training_log.jsonl \
    exp3/training_log.jsonl \
    --labels "Baseline" "Experiment 2" "Experiment 3"
```

## File Locations Summary

```
project/
├── dqn_agent.py              # Main training script
├── training_monitor.py       # Monitoring tool
├── tcp.cc                    # NS-3 simulation
├── logs/                     # Default log directory
│   ├── training_log.jsonl    # Episode-by-episode log
│   ├── training_summary.json # Final statistics
│   └── training_progress_*.png  # Plot images
└── models/                   # Saved model checkpoints
    ├── dqn_tcp_episode_*.pth
    └── dqn_tcp_final.pth
```

## Additional Commands

### Quick Statistics Only (No Plots)
```python
from training_monitor import load_training_log
monitor = load_training_log('./logs/training_log.jsonl')
monitor.print_statistics()
# Don't call plot_training_progress()
```

### Export Data for External Analysis
```python
import json
data = []
with open('./logs/training_log.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))

import pandas as pd
df = pd.DataFrame(data)
df.to_csv('training_data.csv', index=False)
```

## Support

For issues or questions:
1. Check that NS-3 simulation is running and connected
2. Verify logs are being created: `ls -la ./logs/`
3. Check for error messages in both terminals
4. Ensure all dependencies are installed
