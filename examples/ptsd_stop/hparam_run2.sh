python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_avg_roba128_rangel40u80_5folds.pkl' --val_folds 1 \
       --do_hparam_tune --n_trials 100 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --dropout 0.05 --output_dropout 0.1 --num_layer 1 --min_epochs 15 --epochs 100 --lr 5e-5 --weight_decay 1e-3 \
       --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_avg_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_avg_folds1_v1 --experiment_name hp_roba128_1Lgru_ln