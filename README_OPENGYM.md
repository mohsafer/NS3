# NS3 OpenGym Integration for TCP Congestion Control with DQN

This repository contains an NS3 simulation integrated with OpenGym to enable Reinforcement Learning (RL) algorithms for TCP congestion control. The code implements a Deep Q-Network (DQN) agent that learns optimal congestion control policies.

## Overview

The simulation creates a dumbbell network topology with TCP flows and integrates with the ns3-gym framework to allow Python-based RL agents to control network behavior in real-time.

### Network Topology

```
 N0----                            ----N5
       |          (p2p)           |
 N1---------N3 <--------> N4-----------N6
       |                          |
 N2----                            ----N7
```

- **TCP flows**: N0→N5, N1→N6
- **UDP flow**: N2→N7
- **Bottleneck link**: N3↔N4

## Features

- **OpenGym Integration**: Real-time interaction between NS3 and Python RL agents
- **DQN Agent**: Deep Q-Network implementation for learning congestion control
- **Observation Space**: 5-dimensional continuous space
  - Current Congestion Window (normalized)
  - Current RTT (normalized)
  - Throughput (normalized)
  - Packet Loss Rate
  - CWND change rate
- **Action Space**: 5 discrete actions
  - 0: Decrease rate by 50%
  - 1: Decrease rate by 25%
  - 2: Maintain current rate
  - 3: Increase rate by 25%
  - 4: Increase rate by 50%
- **Reward Function**: Balances throughput, packet loss, and delay

## Requirements

### NS3 Requirements

1. **NS3 (version 3.30 or later)**
2. **ns3-gym module**
   - Clone from: https://github.com/tkn-tub/ns3-gym
   - Follow installation instructions in the ns3-gym repository

### Python Requirements

```bash
pip install torch numpy gym ns3gym protobuf==3.20.0
```

**Note**: If you encounter protobuf version issues, use `protobuf==3.20.0`

## Installation

### Step 1: Install ns3-gym

```bash
# Clone ns3-gym
git clone https://github.com/tkn-tub/ns3-gym.git

# Install Python interface
cd ns3-gym/py
pip install -e .
```

### Step 2: Copy OpenGym module to NS3

```bash
# Assuming you have NS3 installed in ~/ns-3-dev/
cp -r ns3-gym/src/opengym ~/ns-3-dev/src/

# Reconfigure NS3
cd ~/ns-3-dev
./waf configure --enable-examples --enable-tests
./waf build
```

### Step 3: Place your simulation file

```bash
# Copy tcp.cc to NS3 scratch directory
cp tcp.cc ~/ns-3-dev/scratch/
```

## Usage

### Running the Simulation with OpenGym

You have two options to run the simulation:

#### Option 1: Manual two-terminal approach

**Terminal 1** - Start NS3 simulation:
```bash
cd ~/ns-3-dev
./waf --run "tcp --openGym=1 --simTime=50 --envStepTime=0.1 --openGymPort=5555"
```

**Terminal 2** - Run Python DQN agent:
```bash
cd /path/to/NS3/
python3 dqn_agent.py --port 5555 --episodes 1000
```

#### Option 2: Let Python start the simulation

```bash
python3 simple_train.py
```

### Command Line Arguments

#### NS3 Simulation Arguments

```bash
./waf --run "tcp --openGym=1 \
             --simTime=50 \
             --envStepTime=0.1 \
             --openGymPort=5555 \
             --tcpVariant=TcpNewReno \
             --simSeed=1"
```

- `--openGym`: Enable OpenGym (0=disabled, 1=enabled)
- `--simTime`: Total simulation time in seconds
- `--envStepTime`: RL agent decision interval in seconds
- `--openGymPort`: Port for OpenGym communication
- `--tcpVariant`: TCP variant (TcpNewReno, TcpCubic, TcpVegas, etc.)
- `--simSeed`: Random seed for reproducibility

#### Python DQN Agent Arguments

```bash
python3 dqn_agent.py --port 5555 \
                     --episodes 1000 \
                     --device cpu \
                     --save dqn_model.pth
```

- `--port`: OpenGym port (must match NS3)
- `--episodes`: Number of training episodes
- `--test`: Run in test mode (load existing model)
- `--load`: Load model from file
- `--save`: Save model to file
- `--device`: Use 'cpu' or 'cuda' for training

## DQN Implementation

The DQN agent (`dqn_agent.py`) implements:

- **Neural Network**: 4-layer fully connected network
- **Experience Replay**: Buffer size of 10,000 transitions
- **Target Network**: Updated every 10 episodes
- **Epsilon-Greedy**: Exploration strategy with decay
- **Hyperparameters**:
  - Learning rate: 0.001
  - Discount factor (γ): 0.99
  - Batch size: 64
  - Initial ε: 1.0
  - Final ε: 0.01
  - ε decay: 0.995

## Code Structure

```
.
├── tcp.cc              # NS3 simulation with OpenGym integration
├── dqn_agent.py        # Full DQN implementation
├── simple_train.py     # Simple training script for testing
└── README.md           # This file
```

### Key Components in tcp.cc

1. **TcpOpenGymEnv Class**: OpenGym environment implementation
   - `GetObservation()`: Returns current network state
   - `GetActionSpace()`: Defines available actions
   - `GetReward()`: Calculates reward function
   - `ExecuteActions()`: Applies RL agent's action

2. **Main Function**: Sets up network topology and connects OpenGym

3. **Trace Callbacks**: Connect network events to OpenGym environment

## Training

### Quick Start Training

```bash
# Simple test with random actions
python3 simple_train.py

# Full DQN training
python3 dqn_agent.py --episodes 1000 --save trained_model.pth
```

### Monitor Training Progress

The agent prints statistics every 10 episodes:
```
Episode 10/1000 | Avg Reward: 5.23 | Epsilon: 0.950 | Loss: 0.0234
Episode 20/1000 | Avg Reward: 6.45 | Epsilon: 0.903 | Loss: 0.0198
...
```

### Testing Trained Model

```bash
python3 dqn_agent.py --test --load trained_model.pth
```

## Customization

### Modify Observation Space

Edit the `GetObservation()` method in `TcpOpenGymEnv`:

```cpp
Ptr<OpenGymDataContainer>
TcpOpenGymEnv::GetObservation()
{
  // Add/modify observations here
  box->AddValue(yourNewObservation);
  return box;
}
```

### Modify Reward Function

Edit the `GetReward()` method:

```cpp
float TcpOpenGymEnv::GetReward()
{
  // Customize reward calculation
  float reward = throughput_component - loss_penalty - delay_penalty;
  return reward;
}
```

### Modify Action Space

Edit `GetActionSpace()` and `ExecuteActions()` methods to change available actions.

## Troubleshooting

### Issue: "cannot open source file ns3/opengym-module.h"

**Solution**: This is expected during initial development. The OpenGym module needs to be installed in your NS3 installation. Follow the ns3-gym installation steps above.

### Issue: "Error connecting to NS3"

**Solution**: 
1. Ensure NS3 simulation is running first
2. Check that port numbers match
3. Verify firewall settings

### Issue: "protobuf version error"

**Solution**:
```bash
pip install protobuf==3.20.0
```

### Issue: Simulation hangs or doesn't progress

**Solution**:
1. Check that Python agent is connected and running
2. Verify `envStepTime` is not too small
3. Check NS3 logs for errors

## Performance Tips

1. **Use GPU for training**: Add `--device cuda` if available
2. **Adjust batch size**: Larger batches (128-256) can stabilize training
3. **Tune epsilon decay**: Slower decay allows more exploration
4. **Parallel environments**: Run multiple simulations for faster data collection

## References

- [ns3-gym](https://github.com/tkn-tub/ns3-gym): OpenGym framework for NS3
- [NS3 Documentation](https://www.nsnam.org/documentation/)
- [DQN Paper](https://arxiv.org/abs/1312.5602): Mnih et al., "Playing Atari with Deep Reinforcement Learning"

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{ns3-opengym-tcp,
  author = {Your Name},
  title = {NS3 OpenGym Integration for TCP Congestion Control with DQN},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/ns3-tcp-dqn}
}
```

## License

This project is licensed under the GPL-2.0 License (same as NS3).

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## Contact

For questions or support, please open an issue on GitHub.
