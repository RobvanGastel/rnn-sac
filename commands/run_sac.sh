source ~/miniconda3/etc/profile.d/conda.sh
conda activate meta 

# # Utilize GRU RNN cell
# args=(
#     --env 'Pendulum-v0' \
#     --rnn_cell 'GRU' \
#     --hid 512 \
#     --lr 0.003 \
#     --seed 42 \
#     --epochs 30 \
#     --alpha 0.2 \
#     --batch_size 10 \
#     --exp_name 'gru_sac_pendulum'
# )

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
