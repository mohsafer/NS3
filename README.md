# NS3: Congestion Control Algorithm Comparison with OpenGym RL Integration

This project utilizes the [ns-3 network simulator](https://www.nsnam.org/) to compare the performance of various TCP congestion control algorithms, including **NewReno** and **Vegas**. 

**NEW**: This project now includes integration with **NS3 OpenGym** for Reinforcement Learning-based congestion control using **Deep Q-Networks (DQN)**. The RL agent can learn optimal TCP congestion control policies through interaction with the NS3 simulation.

The simulations are designed to analyze how these algorithms behave under different network conditions, providing insights into their efficiency and responsiveness.

## Network Topology

The simulated network topology is structured as follows:

```
N0---- ----N5
           |
           | (p2p)
           |
N1---------N3 <--------> N4-----------N6
           |
           |
          N2
```

- **N0 to N6** represent network nodes.
- **N3** acts as a central router connecting different segments.
- **p2p** denotes point-to-point links between nodes.

## Features

- **TCP Algorithm Comparison**: Evaluate and compare the performance of TCP NewReno and TCP Vegas.
- **Custom Topology**: A specifically designed network topology to test various scenarios.
- **Performance Metrics**: Analyze throughput, latency, and packet loss for each algorithm.

## Getting Started

### Prerequisites

- **ns-3**: Ensure that ns-3 is installed on your system. You can download it from the [official website](https://www.nsnam.org/).

### Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/mohsafer/NS3.git
   ```

2. **Navigate to the Project Directory**:

   ```bash
   cd NS3
   ```

3. **Build the Project**:

   Assuming you have ns-3 set up correctly:

   ```bash
   ./waf build
   ```

### Running Simulations

To run the simulation:

```bash
./waf --run scratch/tcp
```

Ensure that the `tcp.cc` file is located in the `scratch/` directory of your ns-3 installation.

## Results

The simulation outputs will provide performance metrics for each congestion control algorithm. Analyze these results to determine the efficiency and suitability of each algorithm under the simulated network conditions.

## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to fork the repository and submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
