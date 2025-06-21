export CUDA_VISIBLE_DEVICES=1

data_path=exchange_rate.csv
random_seed=2025

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id exchange_96_96 \
  --model L3former \
  --data custom \
  --root_path ./dataset/exchange_rate/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 8 \
  --d_model 128 \
  --d_twff 128 \
  --d_vwff 128 \
  --window_size_list 3,7 \
  --learning_rate 0.0001 \
  --head_dropout 0. \
  --dropout 0. \
  --vwff_dropout 0.5 \
  --mask 1 \
  --use_norm_in_former 0 \
  --use_pooling_init 1 \
  --flatten_mod 0 \
  --init_residual_weight_list 1.0,1.0,1.5 \
  --use_scheduler 1 \
  --e_layers 1  >logs/Exchange_${random_seed}_96_96.log
#  --e_layers 1


python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id exchange_96_192 \
  --model L3former \
  --data custom \
  --root_path ./dataset/exchange_rate/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 8 \
  --d_model 128 \
  --d_twff 128 \
  --d_vwff 128 \
  --window_size_list 3,7 \
  --learning_rate 0.0001 \
  --head_dropout 0. \
  --dropout 0. \
  --vwff_dropout 0.5 \
  --mask 1 \
  --use_norm_in_former 0 \
  --use_pooling_init 1 \
  --flatten_mod 0 \
  --init_residual_weight_list 1.0,1.0,1.5 \
  --use_scheduler 1 \
  --e_layers 1  >logs/Exchange_${random_seed}_96_192.log
#  --e_layers 1

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id exchange_96_336 \
  --model L3former \
  --data custom \
  --root_path ./dataset/exchange_rate/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 8 \
  --d_model 128 \
  --d_twff 128 \
  --d_vwff 128 \
  --window_size_list 3,7 \
  --learning_rate 0.0001 \
  --head_dropout 0. \
  --dropout 0. \
  --vwff_dropout 0.5 \
  --mask 1 \
  --use_norm_in_former 0 \
  --use_pooling_init 1 \
  --flatten_mod 0 \
  --init_residual_weight_list 1.0,1.0,1.5 \
  --use_scheduler 1 \
  --e_layers 1  >logs/Exchange_${random_seed}_96_336.log
#  --e_layers 1

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id exchange_96_96 \
  --model L3former \
  --data custom \
  --root_path ./dataset/exchange_rate/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 8 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 256 \
  --window_size_list 3,7 \
  --learning_rate 0.0001 \
  --dropout 0.2 \
  --vwff_dropout 0.1 \
  --revin 0 \
  --mask 1 \
  --use_norm_in_former 0 \
  --use_pooling_init 1 \
  --flatten_mod 0 \
  --init_residual_weight_list 1.0,1.0,1.5 \
  --e_layers 1  >logs/Exchange_${random_seed}_96_720.log
#  --e_layers 1