python run_PCL11_ans.py --data_file '/data/avirinchipur/EMI/datadicts/voicemails_PCL11_minmax_ans_avg_roba128_rangel3u14_rand2.pkl' --do_train \
       --train_batch_size 64 --val_batch_size 64 --num_workers 4 --dropout 0.05 --output_dropout 0.1 --num_layer 2 --min_epochs 200 --epochs 1000 --lr 5e-5  \
       --input_size 128 --hidden_size 64 \
       --early_stopping_patience 0 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/voicemails/PCL11_minmax_ans_avg_roba128' --weight_decay 1e-3 \
       --workspace EMI --project_name voicemails_PCL11_minmax_ans_avg --experiment_name roba128_hidden64_lr5e-5_bs64_2layers_rand2_newdry9
