# Tensorboard commands

# Localhosting of experiment
tensorboard --logdir=data/local/experiment

# Hosting tensorboard results
tensorboard dev upload -logdir=data/local/experiment \
    --name "Comparisons of MAML and RL2 on ML1 and ML10" \
    --description "Comparison of base implementations in Garage"
