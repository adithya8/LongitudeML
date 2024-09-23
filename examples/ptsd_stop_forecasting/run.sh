lr=0.0038440298490446697
weight_decay=3.729612024794069e-05
output_dropout=0.42685551979656755

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_1day_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 0 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_1d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_1d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold0

lr=0.000498075939384542
weight_decay=1.7063350559026258e-05
output_dropout=0.2781535779174923

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_1day_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 1 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_1d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_1d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold1

lr=0.003649724813432039
weight_decay=1.0665810289482508e-05
output_dropout=0.4887072523599267

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_1day_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 2 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_1d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_1d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold2

lr=0.008942278225458277
weight_decay=9.769241485999559e-05
output_dropout=0.05271797329143185

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_1day_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 3 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_1d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_1d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold3


lr=0.0012851574398024769
weight_decay=0.005378633494497839
output_dropout=0.029737305773107816

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_1day_ahead_roba128_rangel40u80_5folds.pkl' --do_train \
       --do_train --val_folds 4 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --lr $lr --weight_decay $weight_decay --dropout 0.05 --output_dropout $output_dropout  \
       --min_epochs 15 --epochs 100 --num_layer 1 --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_1d_ahead_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_1d_ahead_v1 --experiment_name roba128_1Lgru_ln_fold4
