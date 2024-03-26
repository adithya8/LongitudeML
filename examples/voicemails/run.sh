python run_PCL11_ans.py --data_file '/data/avirinchipur/EMI/datadicts/voicemails_PCL11_minmax_ans_avg_roba128_rangel3u14.pkl' --do_train \
       --train_batch_size 64 --val_batch_size 64 --num_workers 4 --dropout 0.05 --output_dropout 0.4113924122858908 --num_layer 1 --min_epochs 200 --epochs 1000 --lr 0.0008994008363411  \
       --input_size 64 --hidden_size 32 \
       --early_stopping_patience 0 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/voicemails/PCL11_minmax_ans_avg_roba64' --weight_decay 0.0125260625776489 \
       --workspace EMI --project_name voicemails_PCL11_minmax_ans_avg_hparams --experiment_name roba64_bs64_trial50_run_nomask_0infill_dcopy