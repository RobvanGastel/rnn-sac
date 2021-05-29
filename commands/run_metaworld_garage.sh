source ~/miniconda3/etc/profile.d/conda.sh
conda activate meta

python ./rnn-sac/experiments/garage/MAML_TRPO_ML1.py --epochs 300 
python ./rnn-sac/experiments/garage/MAML_TRPO_ML10.py --epochs 300
# To expensive to compute locally
# python ./rnn-sac/experiments/garage/MAML_TRPO_ML45.py --epochs 300

python ./rnn-sac/experiments/garage/RL2_PPO_ML1.py --epochs 300
python ./rnn-sac/experiments/garage/RL2_PPO_ML10.py --epochs 300
# To expensive to compute locally
# python ./rnn-sac/experiments/garage/RL2_PPO_ML45.py --epochs 300