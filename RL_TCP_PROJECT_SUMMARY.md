# NS3 OpenGym RL TCP Project v0.1.10 

## Overview
 NS3 TCP congestion control simulation has been prepared for Reinforcement Learning using the OpenGym module and a DQN (Deep Q-Network) agent.

## Modified Files

### 1. tcp.cc (Modified)
**Location**: `/users/mosafer/NS3/tcp.cc`

**Changes Made**:
- ✅ modified`#include "ns3/opengym-module.h"` for OpenGym support
- ✅ Created `TcpOpenGymEnv` class implementing the OpenGym interface
  - Observation space: 5D continuous (CWND, RTT, throughput, packet loss, CWND rate)
  - Action space: 5 discrete actions (rate adjustments)
  - Reward function: throughput - 10×loss - 2×RTT
- ✅ modified command-line parameters for OpenGym configuration
- ✅ Integrated OpenGym environment with TCP socket traces
- ✅ modified random seed support for reproducibility

**Note**: Will show compile error until OpenGym module is installed in NS3.

## New Files Created

### 2. dqn_agent.py (NEW)
**Location**: `/users/mosafer/NS3/dqn_agent.py`

**Purpose**: Complete DQN implementation for learning TCP congestion control

**Features**:
- Deep Q-Network with 4-layer neural network (128 hidden units)
- Experience replay buffer (10,000 capacity)
- Target network with periodic updates
- Epsilon-greedy exploration with decay
- Model save/load functionality
- Training and testing modes

**Usage**:
```bash
# Training
python3 dqn_agent.py --episodes 1000 --save model.pth

# Testing
python3 dqn_agent.py --test --load model.pth
```

### 3. simple_train.py (NEW)
**Location**: `/users/mosafer/NS3/simple_train.py`

**Purpose**: Quick test script with random actions

**Usage**:
```bash
python3 simple_train.py
```

### 4. training_monitor.py (NEW)
**Location**: `/users/mosafer/NS3/training_monitor.py`

**Purpose**: Visualize and monitor training progress

**Features**:
- Plot rewards, losses, epsilon, episode lengths
- Moving averages for trend analysis
- Statistics summary
- Save plots to file

**Usage**:
```bash
python3 training_monitor.py --log-file ./logs/training_log.jsonl
```

### 5. README_OPENGYM.md (NEW)
**Location**: `/users/mosafer/NS3/README_OPENGYM.md`

**Purpose**: Comprehensive documentation

**Contains**:
- Installation instructions
- Usage examples
- Troubleshooting guide
- Customization guide
- Architecture explanation

### 6. CODE_REVIEW.md (NEW)
**Location**: `/users/mosafer/NS3/CODE_REVIEW.md`

**Purpose**: Detailed code review and technical documentation

**Contains**:
- All modifications explained
- Communication flow diagrams
- Customization examples
- Expected behavior
- Testing checklist

### 7. requirements.txt (NEW)
**Location**: `/users/mosafer/NS3/requirements.txt`

**Purpose**: Python dependencies

**Packages**:
- torch (PyTorch for DQN)
- numpy (numerical operations)
- gym (RL interface)
- protobuf==3.20.0 (communication)
- matplotlib (visualization)
- tensorboard (optional logging)

**Usage**:
```bash
pip3 install -r requirements.txt
```

### 8. setup_check.sh (NEW)
**Location**: `/users/mosafer/NS3/setup_check.sh`

**Purpose**: Verify installation and setup

**Checks**:
- Python and pip installation
- Required packages
- NS3 installation
- OpenGym module presence

**Usage**:
```bash
chmod +x setup_check.sh
./setup_check.sh
```

## File Structure

```
/users/mosafer/NS3/
├── tcp.cc                  # Modified NS3 simulation with OpenGym
├── dqn_agent.py           # Full DQN implementation
├── simple_train.py        # Quick test script
├── training_monitor.py    # Training visualization
├── requirements.txt       # Python dependencies
├── setup_check.sh         # Setup verification script
├── README_OPENGYM.md      # Comprehensive documentation
├── CODE_REVIEW.md         # Technical code review
└── README.md              # Original README (if exists)
```

## Quick Start Guide

### Step 1: Install Dependencies
```bash
# Python packages
pip3 install -r requirements.txt

# NS3 OpenGym module
git clone https://github.com/tkn-tub/ns3-gym.git
cd ns3-gym/py && pip install -e .
cp -r ../src/opengym ~/ns-3-dev/src/
```

### Step 2: Build NS3
```bash
cd ~/ns-3-dev
cp /users/mosafer/NS3/tcp.cc scratch/
./waf configure --enable-examples
./waf build
```

### Step 3: Test Setup
```bash
cd /users/mosafer/NS3
./setup_check.sh
```

### Step 4: Run Simulation

**Terminal 1** - NS3:
```bash
cd ~/ns-3-dev
./waf --run "tcp --openGym=1 --simTime=50"
```

**Terminal 2** - Python Agent:
```bash
cd /users/mosafer/NS3
python3 simple_train.py  # For testing
# OR
python3 dqn_agent.py --episodes 1000  # For training
```

## Key Features Implemented

### ✅ OpenGym Integration
- Real-time bidirectional communication between NS3 and Python
- ZMQ-based socket communication
- Configurable via command-line parameters

### ✅ Observation Space (5D)
1. **Congestion Window** (normalized to [0,1])
2. **Round-Trip Time** (normalized)
3. **Throughput** (normalized to max 10 Mbps)
4. **Packet Loss Rate** (0-1)
5. **CWND Change Rate** (normalized to [0,1])

### ✅ Action Space (5 discrete actions)
0. Decrease rate by 50%
1. Decrease rate by 25%
2. Maintain current rate
3. Increase rate by 25%
4. Increase rate by 50%

### ✅ Reward Function
```
Reward = Throughput(Mbps) - 10×LossRate - 2×RTT(sec)
```

### ✅ DQN Features
- Experience replay
- Target network
- Epsilon-greedy exploration
- Batch training
- Model persistence

## Expected Workflow

1. **Development Phase** (Current)
   - Install NS3 and OpenGym module
   - Compile tcp.cc with OpenGym support
   - Test connection with simple_train.py

2. **Training Phase**
   - Run NS3 simulation
   - Train DQN agent for 1000+ episodes
   - Monitor progress with training_monitor.py
   - Save best models

3. **Evaluation Phase**
   - Test trained models
   - Compare with baseline TCP algorithms
   - Analyze performance metrics
   - Fine-tune hyperparameters

4. **Deployment Phase**
   - Deploy best model
   - Continuous monitoring
   - Online learning (optional)

## Important Notes

⚠️ **Compilation**: tcp.cc will show errors until OpenGym module is installed in NS3

⚠️ **Synchronization**: NS3 waits for Python agent - both must be running

⚠️ **Port**: Default port 5555 must be free and match in both NS3 and Python

⚠️ **Step Time**: Balance between control granularity and computational overhead

✅ **Random Seed**: Use consistent seeds for reproducible experiments

✅ **Logging**: All training data saved to logs/ directory

## Troubleshooting

### "cannot open source file ns3/opengym-module.h"
→ Install OpenGym module in NS3 (see README_OPENGYM.md)

### "Error connecting to NS3"
→ Ensure NS3 is running first and ports match

### "protobuf version error"
→ Install: `pip install protobuf==3.20.0`

### Simulation hangs
→ Check Python agent is running and connected

## Next Steps

1. ✅ Review CODE_REVIEW.md for technical details
2. ✅ Read README_OPENGYM.md for installation guide
3. ⬜ Install NS3 and OpenGym module
4. ⬜ Run setup_check.sh to verify
5. ⬜ Test with simple_train.py
6. ⬜ Begin DQN training
7. ⬜ Monitor and optimize

## Additional Resources

- **NS3 OpenGym**: https://github.com/tkn-tub/ns3-gym
- **NS3 Documentation**: https://www.nsnam.org/documentation/
- **DQN Paper**: https://arxiv.org/abs/1312.5602
- **TCP DQN Paper**:  https://arxiv.org/abs/2508.01047

## Support

For questions or issues:
1. Check README_OPENGYM.md troubleshooting section
2. Review CODE_REVIEW.md for implementation details
3. Run setup_check.sh to diagnose setup issues

---

**Status**: ✅ Code prepared and ready for OpenGym integration
**Next**: Install NS3 OpenGym module and begin testing
