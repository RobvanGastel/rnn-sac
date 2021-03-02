# pwd
# ~/Git/meta-learners
conda create --name meta python=3.7
conda activate meta

cd ..
git clone https://github.com/rlworkgroup/garage.git
cd garage


# pip dependencies
# pip install bsuite

# In the correct dir
pip install -e .['dev','all']
pip install gym-bandits

# conda dependencies
# Care for GPU cuda should also be installed,
Y | conda install pytorch torchvision torchaudio -c pytorch
Y | conda install -c conda-forge gym
Y | conda install -c conda-forge jupyterlab
# Y | conda install -c conda-forge matplotlib
# Y | conda install -c conda-forge pandas
# Y | conda install -c conda-forge tensorflow