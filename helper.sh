#!/bin/bash
# Helper script for common NS3 OpenGym DQN operations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
NS3_DIR="$HOME/ns-3-dev"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENGYM_PORT=5555

# Functions
print_help() {
    echo "NS3 OpenGym DQN Helper Script"
    echo ""
    echo "Usage: ./helper.sh [command]"
    echo ""
    echo "Commands:"
    echo "  setup          - Check setup and install dependencies"
    echo "  install-deps   - Install Python dependencies"
    echo "  build          - Build NS3 with tcp.cc"
    echo "  run-sim        - Run NS3 simulation with OpenGym"
    echo "  run-random     - Run simple random agent test"
    echo "  train          - Train DQN agent"
    echo "  test           - Test trained DQN model"
    echo "  monitor        - Show training progress"
    echo "  clean          - Clean logs and temporary files"
    echo "  help           - Show this help message"
    echo ""
}

setup() {
    echo -e "${GREEN}Running setup check...${NC}"
    bash "$SCRIPT_DIR/setup_check.sh"
}

install_deps() {
    echo -e "${GREEN}Installing Python dependencies...${NC}"
    pip3 install -r "$SCRIPT_DIR/requirements.txt"
    echo -e "${GREEN}✅ Dependencies installed${NC}"
}

build_ns3() {
    echo -e "${GREEN}Building NS3 simulation...${NC}"
    
    if [ ! -d "$NS3_DIR" ]; then
        echo -e "${RED}❌ NS3 directory not found: $NS3_DIR${NC}"
        echo "Please set NS3_DIR in this script or install NS3"
        exit 1
    fi
    
    # Copy tcp.cc to scratch
    echo "Copying tcp.cc to NS3 scratch directory..."
    cp "$SCRIPT_DIR/tcp.cc" "$NS3_DIR/scratch/"
    
    # Build
    cd "$NS3_DIR"
    echo "Configuring NS3..."
    ./waf configure --enable-examples --enable-tests
    
    echo "Building NS3..."
    ./waf build
    
    echo -e "${GREEN}✅ Build complete${NC}"
}

run_simulation() {
    echo -e "${GREEN}Starting NS3 simulation with OpenGym...${NC}"
    
    if [ ! -d "$NS3_DIR" ]; then
        echo -e "${RED}❌ NS3 directory not found: $NS3_DIR${NC}"
        exit 1
    fi
    
    cd "$NS3_DIR"
    ./waf --run "tcp --openGym=1 --simTime=50 --envStepTime=0.1 --openGymPort=$OPENGYM_PORT"
}

run_random_agent() {
    echo -e "${GREEN}Running simple random agent test...${NC}"
    echo "Make sure NS3 simulation is running in another terminal!"
    echo ""
    sleep 2
    
    cd "$SCRIPT_DIR"
    python3 simple_train.py
}

train_dqn() {
    echo -e "${GREEN}Training DQN agent...${NC}"
    echo "Make sure NS3 simulation is running in another terminal!"
    echo ""
    
    # Get number of episodes
    read -p "Number of episodes (default: 1000): " EPISODES
    EPISODES=${EPISODES:-1000}
    
    # Get save path
    read -p "Model save path (default: dqn_model.pth): " SAVE_PATH
    SAVE_PATH=${SAVE_PATH:-dqn_model.pth}
    
    cd "$SCRIPT_DIR"
    python3 dqn_agent.py --episodes "$EPISODES" --save "$SAVE_PATH" --port "$OPENGYM_PORT"
}

test_dqn() {
    echo -e "${GREEN}Testing DQN agent...${NC}"
    echo "Make sure NS3 simulation is running in another terminal!"
    echo ""
    
    # Get model path
    read -p "Model load path (default: dqn_model.pth): " LOAD_PATH
    LOAD_PATH=${LOAD_PATH:-dqn_model.pth}
    
    if [ ! -f "$LOAD_PATH" ]; then
        echo -e "${RED}❌ Model file not found: $LOAD_PATH${NC}"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    python3 dqn_agent.py --test --load "$LOAD_PATH" --port "$OPENGYM_PORT"
}

monitor_training() {
    echo -e "${GREEN}Showing training progress...${NC}"
    
    LOG_FILE="$SCRIPT_DIR/logs/training_log.jsonl"
    
    if [ ! -f "$LOG_FILE" ]; then
        echo -e "${YELLOW}⚠️  No training log found: $LOG_FILE${NC}"
        echo "Train the agent first to generate logs"
        exit 1
    fi
    
    cd "$SCRIPT_DIR"
    python3 training_monitor.py --log-file "$LOG_FILE"
}

clean() {
    echo -e "${YELLOW}Cleaning logs and temporary files...${NC}"
    
    read -p "This will delete all logs and temporary files. Continue? (y/N): " CONFIRM
    if [ "$CONFIRM" != "y" ] && [ "$CONFIRM" != "Y" ]; then
        echo "Cancelled"
        exit 0
    fi
    
    cd "$SCRIPT_DIR"
    
    # Remove logs
    if [ -d "logs" ]; then
        rm -rf logs
        echo "✅ Removed logs/"
    fi
    
    # Remove model files
    if ls *.pth 1> /dev/null 2>&1; then
        rm -f *.pth
        echo "✅ Removed .pth model files"
    fi
    
    # Remove Python cache
    if [ -d "__pycache__" ]; then
        rm -rf __pycache__
        echo "✅ Removed __pycache__/"
    fi
    
    # Remove plots
    if ls *.png 1> /dev/null 2>&1; then
        rm -f *.png
        echo "✅ Removed .png plot files"
    fi
    
    echo -e "${GREEN}✅ Cleanup complete${NC}"
}

# Main script
case "${1:-help}" in
    setup)
        setup
        ;;
    install-deps)
        install_deps
        ;;
    build)
        build_ns3
        ;;
    run-sim)
        run_simulation
        ;;
    run-random)
        run_random_agent
        ;;
    train)
        train_dqn
        ;;
    test)
        test_dqn
        ;;
    monitor)
        monitor_training
        ;;
    clean)
        clean
        ;;
    help|*)
        print_help
        ;;
esac
