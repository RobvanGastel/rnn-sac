source ~/miniconda3/etc/profile.d/conda.sh
conda activate meta 

# Default SAC implementation by OpenAI
args=(
    --env 'push-v2' \
    --rnn_cell 'GRU' \
    --meta_learning \
    --hid 256 \
    --lr 0.003 \
    --seed 42 \
    --epochs 100 \
    --alpha 0.2 \
    --batch_size 5 \
    --exp_name 'rl2_gru_sac_push'
)

python ./main.py "${args[@]}"

