# pwd
Y | conda create --name rl python=3.7
conda activate rl

# In the correct dir
pip install box2d-py

# When running gridworld
pip install gym_minigrid

# For meta-agents
pip install joblib
Y | conda install -c anaconda mpi4py
Y | conda install -c conda-forge seaborn==0.8.1

# conda dependencies
# Care for GPU cuda should also be installed,
Y | conda install pytorch torchvision torchaudio -c pytorch
Y | conda install -c conda-forge gym[all]
Y | conda install -c conda-forge jupyterlab

pip install -U mujoco_py==2.0.2.5
