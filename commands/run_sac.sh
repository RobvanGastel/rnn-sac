source ~/miniconda3/etc/profile.d/conda.sh
conda activate meta 

# Default SAC implementation by OpenAI
args=(
    --env 'Pendulum-v0' \
    --rnn_cell 'MLP' \
    --hid 256 \
    --lr 0.003 \
    --seed 42 \
    --epochs 30 \
    --batch_size 100 \
    --exp_name 'mlp_sac_pendulum'
)

python ./main.py "${args[@]}"

# Utilize LSTM RNN cell
args_lstm=(
    --env 'Pendulum-v0' \
    --rnn_cell 'LSTM' \
    --hid 256 \
    --lr 0.003 \
    --seed 42 \
    --epochs 30 \
    --alpha 0.2 \
    --batch_size 5 \
    --exp_name 'lstm_sac_pendulum'
)

python ./main.py "${args_lstm[@]}"


# # Utilize GRU RNN cell
args_gru=(
    --env 'Pendulum-v0' \
    --rnn_cell 'GRU' \
    --hid 256 \
    --lr 0.003 \
    --seed 42 \
    --epochs 30 \
    --alpha 0.2 \
    --batch_size 5 \
    --exp_name 'gru_sac_pendulum'
)

python ./main.py "${args_gru[@]}"
