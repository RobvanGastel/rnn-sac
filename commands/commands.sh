# Tensorboard commands

# Localhosting of experiment
tensorboard --logdir=data/local/experiment

# Hosting tensorboard results
tensorboard dev upload --logdir data/log \
    --name "(optional) My latest experiment" \
    --description "(optional) Simple comparison of several hyperparameters"
