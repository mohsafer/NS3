#!/usr/bin/env python3
"""
Simple DQN training script for NS3 OpenGym TCP Congestion Control
This is a minimal example to get started quickly.
"""

import gym
import numpy as np
from ns3gym import ns3env

def main():
    print("Starting NS3 OpenGym DQN Training")
    print("="*50)
    
    # Environment parameters
    port = 5555
    simTime = 50  # seconds
    
    # Start NS3 simulation with OpenGym
    env = ns3env.Ns3Env(port=port, startSim=True, simSeed=0, simArgs={
        "--openGym": "1",
        "--simTime": str(simTime),
        "--envStepTime": "0.1"
    })
    
    # Get observation and action space
    ob_space = env.observation_space
    ac_space = env.action_space
    
    print(f"Observation space: {ob_space}")
    print(f"Action space: {ac_space}")
    
    # Simple random agent for testing
    stepIdx = 0
    currIt = 0
    
    try:
        obs = env.reset()
        print(f"Initial observation: {obs}")
        
        while True:
            # Random action selection (replace with DQN)
            action = ac_space.sample()
            
            print(f"Step: {stepIdx}, Action: {action}")
            
            obs, reward, done, info = env.step(action)
            print(f"Observation: {obs}")
            print(f"Reward: {reward}")
            print(f"Info: {info}")
            
            stepIdx += 1
            
            if done:
                print("Episode finished!")
                currIt += 1
                
                if currIt == 3:  # Run 3 episodes
                    break
                    
                obs = env.reset()
                stepIdx = 0
    
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    
    finally:
        env.close()
        print("Environment closed")


if __name__ == '__main__':
    main()
