python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_avg_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 0 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr 0.000485544384946633 --weight_decay 0.0004987033039464467 --dropout 0.05 --output_dropout 0.32572686926645467 \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_avg_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_avg_v1.f --experiment_name roba128_1Lgru_ln_fold0

# python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_avg_roba128_rangel40u80_5folds.pkl' --do_train \
#        --do_train --val_folds 1 \
#        --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
#        --lr 0.001198368880341542 --weight_decay 0.0038192782727480864 --dropout 0.05 --output_dropout 0.3579773486707952 \
#        --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
#        --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
#        --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_avg_roba128' \
#        --workspace EMI --project_name ptsd_stop_PCL_avg_v1.f --experiment_name roba128_1Lgru_ln_fold1

# python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_avg_roba128_rangel40u80_5folds.pkl' --do_train \
#        --do_train --val_folds 2 \
#        --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
#        --lr 0.0021558725283466786 --weight_decay 1.4968399579055085e-05 --dropout 0.05 --output_dropout 0.4500345362695106 \
#        --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
#        --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
#        --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_avg_roba128' \
#        --workspace EMI --project_name ptsd_stop_PCL_avg_v1.f --experiment_name roba128_1Lgru_ln_fold2

# python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_avg_roba128_rangel40u80_5folds.pkl' --do_train \
#        --do_train --val_folds 3 \
#        --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
#        --lr 0.003059399748257763 --weight_decay  1.852957919035083e-05 --dropout 0.05 --output_dropout 0.4730628630266869  \
#        --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
#        --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
#        --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_avg_roba128' \
#        --workspace EMI --project_name ptsd_stop_PCL_avg_v1.f --experiment_name roba128_1Lgru_ln_fold3


# python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_avg_roba128_rangel40u80_5folds.pkl' --do_train \
#        --do_train --val_folds 4 \
#        --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
#        --lr 0.0008089124183366347 --weight_decay 0.0033639820236422715 --dropout 0.05 --output_dropout 0.46225824746862015  \
#        --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
#        --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
#        --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_avg_roba128' \
#        --workspace EMI --project_name ptsd_stop_PCL_avg_v1.f --experiment_name roba128_1Lgru_ln_fold4