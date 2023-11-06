python run_PCL11_ans.py --data_file '/data/avirinchipur/EMI/datadicts/voicemails_PCL11_minmax_ans_avg_roba128_rangel3u14.pkl' --do_hparam_tune --n_trials 100 \
       --train_batch_size 64 --val_batch_size 64 --num_workers 4 --dropout 0.05 --output_dropout 0.1 --num_layer 1 --min_epochs 300 --epochs 1000 --lr 5e-5  \
       --input_size 128 --hidden_size 64 \
       --early_stopping_patience 0 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/voicemails/PCL11_minmax_ans_avg_roba128' --weight_decay 1e-3 \
       --workspace EMI --project_name voicemails_PCL11_minmax_ans_avg_hparams --experiment_name hparam4_roba128_bs64