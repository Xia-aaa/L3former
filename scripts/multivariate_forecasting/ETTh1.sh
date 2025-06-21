export CUDA_VISIBLE_DEVICES=1

data_path=ETTh1.csv
random_seed=2025


python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ETTh1_96_96 \
  --model L3former \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 7 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 128 \
  --batch_size 128 \
  --window_size_list 3,7,15,27 \
  --learning_rate 0.001 \
  --init_residual_weight_list 0.5,0.5,1.0 \
  --use_scheduler 1 \
  --e_layers 1  >logs/ETTh1_${random_seed}_96_96.log
#  --e_layers 1

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ETTh1_96_192 \
  --model L3former \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 7 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 128 \
  --batch_size 128 \
  --window_size_list 3,7,15,27 \
  --learning_rate 0.001 \
  --init_residual_weight_list 0.5,0.5,1.0 \
  --use_scheduler 1 \
  --e_layers 1  >logs/ETTh1_${random_seed}_96_192.log
#  --e_layers 1

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ETTh1_96_336 \
  --model L3former \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 7 \
  --d_model 512 \
  --d_twff 512 \
  --d_vwff 128 \
  --batch_size 128 \
  --window_size_list 3,7,15,27 \
  --learning_rate 0.001 \
  --vwff_dropout 0.5 \
  --init_residual_weight_list 0.5,0.5,1.0 \
  --use_scheduler 1 \
  --e_layers 1
#  --e_layers 1  >logs/ETTh1_${random_seed}_96_336.log
#  --e_layers 1

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id ETTh1_96_720 \
  --model L3former \
  --data ETTh1 \
  --root_path ./dataset/ETT-small/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 7 \
  --d_model 256 \
  --d_twff 256 \
  --d_vwff 128 \
  --batch_size 128 \
  --window_size_list 3,7,15,27 \
  --learning_rate 0.001 \
  --vwff_dropout 0.5 \
  --init_residual_weight_list 0.5,0.5,1.0 \
  --use_scheduler 1 \
  --e_layers 1
#  --e_layers 1  >logs/ETTh1_${random_seed}_96_720.log
#  --e_layers 1


