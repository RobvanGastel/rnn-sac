conda activate meta

# python ./experiments/metaworld/MAML_TRPO_ML1.py --epochs 300 
python ./experiments/metaworld/MAML_TRPO_ML10.py --epochs 300
python ./experiments/metaworld/MAML_TRPO_ML45.py --epochs 300

# python ./experiments/metaworld/RL2_PPO_ML1.py --epochs 300
# python ./experiments/metaworld/RL2_PPO_ML10.py --epochs 300
python ./experiments/metaworld/RL2_PPO_ML45.py --epochs 300