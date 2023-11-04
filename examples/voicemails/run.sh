python run_PCL11_ans.py --data_file '/data/avirinchipur/EMI/datadicts/voicemails_PCL11_minmax_ans_avg_roba128_rangel3u14_rand2.pkl' --do_train \
       --train_batch_size 64 --val_batch_size 64 --num_workers 4 --output_dropout 0.0 --num_layer 1 --min_epochs 200 --epochs 2000 --lr 5e-3  \
       --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/voicemails/PCL11_minmax_ans_avg_roba128' --weight_decay 1e-1 \
       --workspace EMI --project_name voicemails_PCL11_minmax_ans_avg --experiment_name roba128_hidden64_lr1e-4_bs64_rand2_newdry3
