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
pip install -e . # ['dev','all'], given by the devs but doesn't work
pip install box2d-py

# conda dependencies
# Care for GPU cuda should also be installed,
Y | conda install pytorch torchvision torchaudio -c pytorch
Y | conda install -c conda-forge gym[all]
Y | conda install -c conda-forge jupyterlab

pip install -U mujoco_py==2.0.2.5
pip install git+https://github.com/rlworkgroup/metaworld.git@master#egg=metaworld
