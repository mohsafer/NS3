#!/bin/bash
# Setup script for NS3 OpenGym DQN environment

echo "=================================="
echo "NS3 OpenGym DQN Setup Checker"
echo "=================================="
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "❌ Python3 not found. Please install Python 3.7 or higher."
    exit 1
fi
echo "✅ Python3 found"
echo ""

# Check pip
echo "Checking pip..."
pip3 --version
if [ $? -ne 0 ]; then
    echo "❌ pip3 not found. Please install pip."
    exit 1
fi
echo "✅ pip3 found"
echo ""

# Check if requirements are installed
echo "Checking Python dependencies..."
MISSING_DEPS=0

# Check torch
python3 -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ PyTorch not installed"
    MISSING_DEPS=1
else
    echo "✅ PyTorch installed"
fi

# Check numpy
python3 -c "import numpy" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ NumPy not installed"
    MISSING_DEPS=1
else
    echo "✅ NumPy installed"
fi

# Check gym
python3 -c "import gym" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Gym not installed"
    MISSING_DEPS=1
else
    echo "✅ Gym installed"
fi

# Check ns3gym
python3 -c "import ns3gym" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  ns3gym not installed (this is normal if not set up yet)"
    echo "   Install with: pip install ns3gym"
else
    echo "✅ ns3gym installed"
fi

echo ""

if [ $MISSING_DEPS -eq 1 ]; then
    echo "Some dependencies are missing. Install them with:"
    echo "  pip3 install -r requirements.txt"
    echo ""
fi

# Check NS3 installation
echo "Checking NS3 installation..."
if [ -d "$HOME/ns-3-dev" ]; then
    echo "✅ NS3 directory found at $HOME/ns-3-dev"
    
    # Check for waf
    if [ -f "$HOME/ns-3-dev/waf" ]; then
        echo "✅ waf build system found"
    else
        echo "❌ waf not found in NS3 directory"
    fi
    
    # Check for opengym module
    if [ -d "$HOME/ns-3-dev/src/opengym" ]; then
        echo "✅ OpenGym module found in NS3"
    else
        echo "⚠️  OpenGym module not found in NS3"
        echo "   Install from: https://github.com/tkn-tub/ns3-gym"
    fi
else
    echo "⚠️  NS3 not found at $HOME/ns-3-dev"
    echo "   Adjust path or install NS3"
fi

echo ""
echo "=================================="
echo "Setup Summary"
echo "=================================="
echo ""
echo "To complete setup:"
echo "1. Install Python dependencies:"
echo "   pip3 install -r requirements.txt"
echo ""
echo "2. Install ns3-gym:"
echo "   git clone https://github.com/tkn-tub/ns3-gym.git"
echo "   cd ns3-gym/py && pip install -e ."
echo ""
echo "3. Copy OpenGym module to NS3:"
echo "   cp -r ns3-gym/src/opengym ~/ns-3-dev/src/"
echo "   cd ~/ns-3-dev && ./waf configure && ./waf build"
echo ""
echo "4. Copy tcp.cc to NS3 scratch directory:"
echo "   cp tcp.cc ~/ns-3-dev/scratch/"
echo ""
echo "5. Build and run:"
echo "   cd ~/ns-3-dev"
echo "   ./waf --run \"tcp --openGym=1\""
echo ""
echo "6. In another terminal, run the DQN agent:"
echo "   python3 dqn_agent.py"
echo ""
