conda create --name meta python=3.7
conda activate meta

# pip dependencies
# pip install bsuite

# In the correct dir
pip install -e .['dev','all']
pip install gym-bandits

# conda dependencies
Y | conda install pytorch torchvision torchaudio -c pytorch
# Y | conda install -c conda-forge tensorflow
Y | conda install -c conda-forge gym
Y | conda install -c conda-forge jupyterlab
# Y | conda install -c conda-forge matplotlib
# Y | conda install -c conda-forge pandas
