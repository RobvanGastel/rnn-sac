# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate temp

# TODO: Adjust the buffer to make this work
# python ./meta-learners/experiments/SAC/RNN_SAC_bipedal.py
# python ./meta-learners/experiments/SAC/RL2_SAC_bipedal.py

# TODO: Wait for license on Mujoco
# python ./meta-learners/experiments/SAC/SAC_fetch.py
# python ./meta-learners/experiments/SAC/RL2_SAC_fetch.py

# python ./meta-learners/experiments/SAC/SAC_pendulum.py
# python ./meta-learners/experiments/SAC/RNN_SAC_pendulum.py 

# Run bipedal tasks
python ./meta-learners/experiments/SAC/RL2_SAC_bipedal.py
python ./meta-learners/experiments/SAC/RNN_SAC_bipedal.py
