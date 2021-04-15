# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate temp

python ./meta-learners/experiments/SAC/SAC_bipedal.py 
python ./meta-learners/experiments/SAC/SAC_pendulum.py
python ./meta-learners/experiments/SAC/RL2_SAC_bipedal.py 
python ./meta-learners/experiments/SAC/LSTM_SAC_pendulum.py 
