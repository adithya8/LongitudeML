lr=0.001641662889508633
weight_decay=0.0018992876846974842
output_dropout=0.324013024009401

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_2days_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 0 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_2d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_2d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold0

lr=0.001727812188052317
weight_decay=0.0021443244631774877
output_dropout=0.12653500678301954

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_2days_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 1 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_2d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_2d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold1

lr=0.00931078173653311
weight_decay=0.0014279578968771022
output_dropout=0.16739758811161806

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_2days_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 2 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_2d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_2d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold2

lr=0.006131440457398056
weight_decay=0.00115972369193188
output_dropout=0.23231935466076276

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_2days_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 3 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_2d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_2d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold3


lr=0.009425999185032471
weight_decay=0.003563584168202269
output_dropout=0.017167230502860287

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_2days_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 4 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_2d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_2d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold4
