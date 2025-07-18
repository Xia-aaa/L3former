export CUDA_VISIBLE_DEVICES=1

data_path=electricity.csv
random_seed=2025

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ECL_96_96 \
  --model L3former \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 321 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 128 \
  --window_size_list 3,7 \
  --learning_rate 0.001 \
  --dropout 0.2 \
  --e_layers 3  >logs/ECL_${random_seed}_96_96.log
#  --e_layers 3

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ECL_96_192 \
  --model L3former \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 321 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 128 \
  --window_size_list 3,7 \
  --learning_rate 0.001 \
  --dropout 0.2 \
  --e_layers 3  >logs/ECL_${random_seed}_96_192.log
#  --e_layers 3

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ECL_96_336 \
  --model L3former \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 321 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 128 \
  --window_size_list 3,7 \
  --learning_rate 0.001 \
  --dropout 0.2 \
  --e_layers 3  >logs/ECL_${random_seed}_96_336.log
#  --e_layers 3

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ECL_96_720 \
  --model L3former \
  --data custom \
  --root_path ./dataset/electricity/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 321 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 512 \   # or d_vwff 128,256
  --window_size_list 3,7 \
  --learning_rate 0.001 \
  --dropout 0.2 \
  --e_layers 3  >logs/ECL_${random_seed}_96_720.log
#  --e_layers 3

