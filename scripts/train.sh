#!/bin/bash
set -e

# ===============================================================================
#                             SUPPORTED ENVIRONMENTS
# ===============================================================================
# DeepMind Control Suite (benchmark='dmc'):
#   acrobot-swingup, cheetah-run, dog-run, dog-stand, dog-trot, dog-walk,
#   finger-turn_hard, fish-swim, hopper-hop, humanoid-run, humanoid-stand, 
#   humanoid-walk, pendulum-swingup, quadruped-run, reacher-hard, walker-run
#
# HumanoidBench (benchmark='hb'):
#   h1-balance_simple-v0, h1-balance_hard-v0, h1-crawl-v0, h1-hurdle-v0, 
#   h1-maze-v0, h1-pole-v0, h1-reach-v0, h1-run-v0, h1-slide-v0, 
#   h1-sit_hard-v0, h1-sit_simple-v0, h1-stair-v0, h1-stand-v0, h1-walk-v0
#
# MyoSuite (benchmark='myo'):
#   myo-key-turn, myo-key-turn-hard, myo-obj-hold, myo-obj-hold-hard, 
#   myo-pen-twirl, myo-pen-twirl-hard, myo-pose, myo-pose-hard, 
#   myo-reach, myo-reach-hard
# ===============================================================================

if [ "$#" -lt 2 ]; then
    echo "Usage: ./scripts/train.sh <benchmark> <env_name> [args...]"
    echo "Example: ./scripts/train.sh dmc cheetah-run"
    exit 1
fi

BENCHMARK=$1
ENV_NAME=$2
shift 2

# Identify Rollout Horizon (H) per task from ICLR paper Appendix Tables
case $ENV_NAME in
    # H = 10 (Most HumanoidBench locomotion)
    h1-crawl-v0|h1-hurdle-v0|h1-maze-v0|h1-pole-v0|h1-reach-v0|h1-run-v0|h1-slide-v0|h1-sit_simple-v0|h1-stair-v0|h1-stand-v0|h1-walk-v0) 
        H=10 ;;
        
    # H = 8
    acrobot-swingup|fish-swim) 
        H=8 ;;
        
    # H = 6 (Hard DMC tasks, balancing HB tasks, complex MyoSuite)
    humanoid-walk|humanoid-run|dog-stand|dog-run|myo-key-turn|h1-balance_simple-v0|h1-balance_hard-v0|h1-sit_hard-v0) 
        H=6 ;;
        
    # H = 4
    dog-walk|dog-trot|myo-obj-hold|myo-reach) 
        H=4 ;;
        
    # H = 2
    quadruped-run|humanoid-stand|myo-pen-twirl|myo-pose) 
        H=2 ;;
        
    # H = 1 (Most easy DMC tasks and MyoSuite Hard variants)
    cheetah-run|finger-turn_hard|hopper-hop|pendulum-swingup|reacher-hard|walker-run|myo-key-turn-hard|myo-obj-hold-hard|myo-pen-twirl-hard|myo-pose-hard|myo-reach-hard) 
        H=1 ;;
        
    *) 
        echo "Warning: Unknown environment '$ENV_NAME'. Falling back to H=1." 
        H=1 ;;
esac

echo "==============================================="
echo "Launching training for:  $ENV_NAME"
echo "Benchmark Suite:         $BENCHMARK"
echo "Rollout Horizon (H):     $H"
echo "==============================================="

# Navigate to project root
cd "$(dirname "$0")/.."

# Check for activated environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Warning: You do not currently have a Python virtual environment activated!"
    echo "Please ensure you have run 'source .test/bin/activate' or equivalent."
    sleep 2
fi

NVCC_PATH=$(python -c "import site; print(site.getsitepackages()[0] + '/nvidia/cuda_nvcc/bin')")
export PATH=$NVCC_PATH:$PATH

# Run the training
python train_parallel.py \
    --benchmark "$BENCHMARK" \
    --env_name "$ENV_NAME" \
    --run_name "WIMLE_${ENV_NAME}_stal4_model5_lat2" \
    --model_H $H \
    --wandb_mode "online" \
    "$@"
