export CUDA_VISIBLE_DEVICES=1


# scheduler
data_path=solar_AL.txt
random_seed=2025


python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id solar_96_96 \
  --model L3former \
  --data Solar \
  --root_path ./dataset/solar/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 96 \
  --enc_in 137 \
  --d_model 256 \
  --d_twff 256 \
  --d_vwff 64 \
  --patience 5 \
  --batch_size 32 \
  --window_size_list 3,7,15,27,43 \
  --learning_rate 0.0005 \
  --use_std_in_revin 0 \
  --affine 1 \
  --init_residual_weight_list 0.5,1.0,1.0 \
  --e_layers 2  >logs/Solar_${random_seed}_96_96.log
#  --e_layers 2

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id solar_96_192 \
  --model L3former \
  --data Solar \
  --root_path ./dataset/solar/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 192 \
  --enc_in 137 \
  --d_model 256 \
  --d_twff 256 \
  --d_vwff 64 \
  --patience 5 \
  --batch_size 32 \
  --window_size_list 3,7,15,27,43 \
  --learning_rate 0.0005 \
  --use_std_in_revin 0 \
  --affine 1 \
  --init_residual_weight_list 0.5,1.0,1.0 \
  --e_layers 2  >logs/Solar_${random_seed}_96_192.log
#  --e_layers 2

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id solar_96_336 \
  --model L3former \
  --data Solar \
  --root_path ./dataset/solar/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 336 \
  --enc_in 137 \
  --d_model 256 \
  --d_twff 256 \
  --d_vwff 64 \
  --patience 5 \
  --batch_size 32 \
  --window_size_list 3,7,15 \
  --learning_rate 0.0005 \
  --use_std_in_revin 0 \
  --affine 1 \
  --init_residual_weight_list 0.5,1.0,1.0 \
  --e_layers 2  >logs/Solar_${random_seed}_96_336.log
#  --e_layers 2

python -u run_for_L3former.py \
  --random_seed $random_seed \
  --model_id solar_96_720 \
  --model L3former \
  --data Solar \
  --root_path ./dataset/solar/ \
  --data_path $data_path \
  --seq_len 96 \
  --pred_len 720 \
  --enc_in 137 \
  --d_model 256 \
  --d_twff 256 \
  --d_vwff 64 \
  --patience 5 \
  --batch_size 32 \
  --window_size_list 3,7,15 \
  --learning_rate 0.0005 \
  --use_std_in_revin 0 \
  --affine 1 \
  --init_residual_weight_list 0.5,1.0,1.0 \
  --e_layers 2  >logs/Solar_${random_seed}_96_720.log
#  --e_layers 2

