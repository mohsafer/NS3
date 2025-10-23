# Code Review: NS3 OpenGym Integration for DQN-based TCP Congestion Control

## Summary

The `tcp.cc` file has been modified to integrate with NS3's OpenGym module, enabling Reinforcement Learning (RL) algorithms to control TCP congestion behavior in real-time. A complete DQN (Deep Q-Network) implementation has been provided in Python.

## Key Modifications to tcp.cc

### 1. Added OpenGym Module Import

```cpp
#include "ns3/opengym-module.h"
```

### 2. Created TcpOpenGymEnv Class

A new class `TcpOpenGymEnv` that inherits from `OpenGymEnv` to implement the OpenGym interface:

#### Key Methods:

- **GetObservationSpace()**: Defines a 5-dimensional continuous observation space
  - Current Congestion Window (normalized to [0,1])
  - Current RTT (normalized)
  - Throughput (normalized)
  - Packet Loss Rate
  - CWND change rate

- **GetActionSpace()**: Defines 5 discrete actions
  - 0: Decrease sending rate by 50%
  - 1: Decrease rate by 25%
  - 2: Maintain current rate
  - 3: Increase rate by 25%
  - 4: Increase rate by 50%

- **GetObservation()**: Returns current network state as normalized values

- **GetReward()**: Calculates reward based on:
  ```
  Reward = Throughput - (10 × PacketLossRate) - (2 × RTT)
  ```
  This balances throughput maximization with loss and delay minimization.

- **ExecuteActions()**: Processes actions from the RL agent

- **State Tracking Methods**:
  - `SetCwnd()`: Updates congestion window measurements
  - `SetRtt()`: Updates RTT measurements
  - `SetThroughput()`: Updates throughput measurements
  - `SetPacketLoss()`: Updates packet loss statistics

### 3. Added Command Line Parameters

New parameters for OpenGym control:

```cpp
bool openGymEnabled = true;      // Enable/disable OpenGym
uint32_t openGymPort = 5555;     // Communication port
double envStepTime = 0.1;        // RL decision interval (seconds)
uint32_t simSeed = 1;            // Random seed
double simTime = 50.0;           // Simulation duration
```

### 4. Modified Main Function

#### Environment Initialization:
```cpp
Ptr<TcpOpenGymEnv> openGymEnv = CreateObject<TcpOpenGymEnv>(simSeed, simTime, openGymPort, envStepTime);
openGymEnv->SetOpenGymInterface(OpenGymInterface::Get(openGymPort));
```

#### Connected Trace Callbacks:
Modified congestion window traces to update the OpenGym environment:
```cpp
if (openGymEnabled && openGymEnv) {
    ns3TcpSocket->TraceConnectWithoutContext("CongestionWindow", 
        MakeBoundCallback(&TcpOpenGymEnv::SetCwnd, openGymEnv));
}
```

#### Scheduled State Updates:
```cpp
Simulator::Schedule(Seconds(envStepTime), 
    &TcpOpenGymEnv::ScheduleNextStateRead, openGymEnv);
```

## New Python Files Created

### 1. dqn_agent.py (Full DQN Implementation)

Complete Deep Q-Network implementation with:

- **DQNNetwork**: 4-layer neural network for Q-value approximation
- **ReplayBuffer**: Experience replay with capacity of 10,000 transitions
- **DQNAgent**: Main agent class with:
  - Epsilon-greedy exploration
  - Target network (updated every 10 episodes)
  - Adam optimizer
  - MSE loss function
  
**Hyperparameters:**
- Learning rate: 0.001
- Discount factor (γ): 0.99
- Epsilon decay: 0.995 (1.0 → 0.01)
- Batch size: 64
- Hidden layer size: 128 neurons

**Features:**
- Model saving/loading
- Training and testing modes
- Periodic checkpoint saving
- Progress monitoring

### 2. simple_train.py (Quick Test Script)

Minimal script for testing the OpenGym connection with random actions.

### 3. README_OPENGYM.md (Comprehensive Documentation)

Complete guide covering:
- Installation instructions
- Usage examples
- Architecture explanation
- Troubleshooting
- Customization guide

### 4. requirements.txt (Python Dependencies)

Lists all required Python packages:
- torch (PyTorch for DQN)
- numpy (numerical operations)
- gym (RL interface)
- protobuf (communication protocol)

### 5. setup_check.sh (Setup Verification Script)

Bash script to verify:
- Python installation
- Required packages
- NS3 installation
- OpenGym module presence

## How It Works

### Communication Flow:

```
┌─────────────┐         ZMQ Socket         ┌─────────────┐
│             │      (Port 5555)           │             │
│  NS3 C++    │◄──────────────────────────►│  Python DQN │
│ Simulation  │                            │    Agent    │
│             │                            │             │
└─────────────┘                            └─────────────┘
      │                                           │
      │ 1. GetObservation()                       │
      │────────────────────────────────────────►  │
      │                                           │
      │ 2. State (CWND, RTT, throughput, etc.)   │
      │◄────────────────────────────────────────  │
      │                                           │
      │                                           │ 3. DQN decides action
      │                                           │
      │ 4. ExecuteActions(action)                 │
      │◄────────────────────────────────────────  │
      │                                           │
      │ 5. Apply action to network               │
      │                                           │
      │ 6. Calculate reward                       │
      │────────────────────────────────────────►  │
      │                                           │
      │                                           │ 7. Update Q-network
      │                                           │
      └───────────────────────────────────────────┘
```

### Training Loop:

1. **NS3 starts** with OpenGym enabled
2. **Python agent connects** to specified port
3. **Every envStepTime seconds**:
   - NS3 sends current network state (observation)
   - Python DQN agent selects action based on Q-values
   - Action is sent back to NS3
   - NS3 executes action (modifies network behavior)
   - NS3 calculates and sends reward
   - Python agent updates Q-network using experience replay
4. **Process repeats** until simulation ends or training completes

## Usage Examples

### Basic Run (Random Agent):
```bash
# Terminal 1: Start NS3
cd ~/ns-3-dev
./waf --run "tcp --openGym=1"

# Terminal 2: Run random agent
cd /users/mosafer/NS3
python3 simple_train.py
```

### Full DQN Training:
```bash
# Terminal 1: NS3 with specific parameters
cd ~/ns-3-dev
./waf --run "tcp --openGym=1 --simTime=50 --envStepTime=0.1 --tcpVariant=TcpCubic"

# Terminal 2: DQN training
cd /users/mosafer/NS3
python3 dqn_agent.py --episodes 1000 --save trained_model.pth
```

### Testing Trained Model:
```bash
python3 dqn_agent.py --test --load trained_model.pth
```

## Customization Guide

### Modify Observation Space:

Edit `GetObservation()` in `tcp.cc`:
```cpp
box->AddValue(yourNewMetric);  // Add new observation
```

Update `state_size` in Python accordingly.

### Modify Action Space:

Edit `GetActionSpace()` and `ExecuteActions()` in `tcp.cc`:
```cpp
uint32_t actionNum = 7;  // Change number of actions
```

### Modify Reward Function:

Edit `GetReward()` in `tcp.cc`:
```cpp
float reward = custom_throughput_metric - custom_loss_penalty;
```

### Tune DQN Hyperparameters:

Edit `DQNAgent.__init__()` in `dqn_agent.py`:
```python
self.gamma = 0.95           # Change discount factor
self.learning_rate = 0.0001 # Change learning rate
self.batch_size = 128       # Change batch size
```

## Expected Behavior

### During Training:

1. **Early episodes**: High exploration (ε ≈ 1.0), random actions, variable rewards
2. **Mid training**: Decreasing exploration, agent starts learning patterns
3. **Late training**: Low exploration (ε → 0.01), consistent policy, higher average rewards

### Convergence Indicators:

- Increasing average reward over episodes
- Decreasing loss values
- More stable throughput/RTT patterns
- Fewer packet losses

## Potential Improvements

1. **Multi-flow coordination**: Control multiple TCP flows simultaneously
2. **Prioritized experience replay**: Sample important transitions more frequently
3. **Dueling DQN**: Separate value and advantage streams
4. **Double DQN**: Reduce overestimation bias
5. **Curriculum learning**: Start with simple scenarios, increase complexity
6. **Transfer learning**: Pre-train on simpler networks, fine-tune on complex ones

## Important Notes

1. **Compilation**: The code will show compile errors for `opengym-module.h` until the OpenGym module is properly installed in NS3
2. **Port conflicts**: Ensure the port (default 5555) is not used by other processes
3. **Synchronization**: NS3 waits for Python agent to respond; ensure agent is running
4. **Random seed**: Use consistent seeds for reproducible experiments
5. **Step time**: Too small `envStepTime` increases overhead; too large reduces control granularity

## Testing Checklist

- [ ] NS3 compiles with OpenGym module
- [ ] Python dependencies installed
- [ ] NS3 simulation starts without errors
- [ ] Python agent connects successfully
- [ ] Observations are received and valid
- [ ] Actions are applied correctly
- [ ] Rewards are calculated properly
- [ ] Training progresses (loss decreases)
- [ ] Models can be saved and loaded
- [ ] Test mode works with trained models

## Next Steps

1. **Install NS3 and OpenGym module** (see README_OPENGYM.md)
2. **Copy tcp.cc to NS3 scratch directory**
3. **Run setup_check.sh** to verify installation
4. **Test with simple_train.py** to verify connection
5. **Begin training with dqn_agent.py**
6. **Monitor and tune hyperparameters**
7. **Evaluate trained model performance**

## Conclusion

The code is now ready for RL-based TCP congestion control using DQN. The integration provides a flexible framework for experimenting with different RL algorithms and reward functions. The modular design allows easy customization of observations, actions, and rewards to explore various congestion control strategies.
