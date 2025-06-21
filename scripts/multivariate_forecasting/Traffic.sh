export CUDA_VISIBLE_DEVICES=1

data_path=traffic.csv
random_seed=2025


python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id traffic_96_96 \
  --model L3former \
  --data custom \
  --root_path ./dataset/traffic/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 862 \
  --d_model 512 \
  --d_twff 512 \
  --use_vwff 0 \
  --train_epochs 20 \
  --patience 5 \
  --batch_size 16 \
  --window_size_list 3,7,15,27,43 \
  --learning_rate 0.001 \
  --lradj 'TST' \
  --init_residual_weight_list 2.0,1.0,1.0 \
  --train_residual_weight 0 \
  --use_amp \
  --e_layers 4  >logs/Traffic_${random_seed}_96_96.log
#  --e_layers 4

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id traffic_96_192 \
  --model L3former \
  --data custom \
  --root_path ./dataset/traffic/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 862 \
  --d_model 512 \
  --d_twff 512 \
  --use_vwff 0 \
  --train_epochs 20 \
  --patience 5 \
  --batch_size 16 \
  --window_size_list 3,7,15,27,43 \
  --learning_rate 0.001 \
  --lradj 'TST' \
  --init_residual_weight_list 2.0,1.0,1.0 \
  --train_residual_weight 0 \
  --use_amp \
  --e_layers 4  >logs/Traffic_${random_seed}_96_192.log
#  --e_layers 4

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id traffic_96_336 \
  --model L3former \
  --data custom \
  --root_path ./dataset/traffic/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 862 \
  --d_model 512 \
  --d_twff 512 \
  --use_vwff 0 \
  --train_epochs 20 \
  --patience 5 \
  --batch_size 16 \
  --window_size_list 3,7,15,27,43 \
  --learning_rate 0.001 \
  --lradj 'TST' \
  --init_residual_weight_list 2.0,1.0,1.0 \
  --train_residual_weight 0 \
  --use_amp \
  --e_layers 4  >logs/Traffic_${random_seed}_96_336.log
#  --e_layers 4

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id traffic_96_720 \
  --model L3former \
  --data custom \
  --root_path ./dataset/traffic/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 862 \
  --d_model 512 \
  --d_twff 512 \
  --use_vwff 0 \
  --train_epochs 20 \
  --patience 5 \
  --batch_size 16 \
  --window_size_list 3,7,15,27,43 \
  --learning_rate 0.001 \
  --lradj 'TST' \
  --init_residual_weight_list 2.0,1.0,1.0 \
  --train_residual_weight 0 \
  --use_amp \
  --e_layers 4  >logs/Traffic_${random_seed}_96_720.log
#  --e_layers 4