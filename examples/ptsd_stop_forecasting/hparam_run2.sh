
# Check if the number of arguments is 1 
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 <fold_num>"
    exit 1
fi

fold_num=$1
project_name='ptsd_stop_PCL_2d_ahead_folds'$fold_num'_v1'
echo "Fold number: $fold_num"
echo "Project name: $project_name"

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_2days_ahead_roba128_rangel40u80_5folds.pkl' --val_folds $fold_num \
       --do_hparam_tune --n_trials 100 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --dropout 0.05 --output_dropout 0.1 --num_layer 1 --min_epochs 15 --epochs 100 --lr 5e-5 --weight_decay 1e-3 \
       --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_future_roba128' \
       --workspace EMI --project_name $project_name --experiment_name hp_roba128_1Lgru_ln


# Best trial from v2 fold 4:
#   Value: 0.5255086421966553
#   Number: 12
#   Params:
#     lr: 0.009425999185032471
#     weight_decay: 0.003563584168202269
#     output_dropout: 0.017167230502860287

# Best trial from v2 fold 3:
#   Value: 0.6283847689628601
#   Number: 95
#   Params:
#     lr: 0.006131440457398056
#     weight_decay: 0.00115972369193188
#     output_dropout: 0.23231935466076276

# Best trial from v2 fold 2:
#   Value: 0.4150431752204895
#   Number: 12
#   Params:
#     lr: 0.00931078173653311
#     weight_decay: 0.0014279578968771022
#     output_dropout: 0.16739758811161806

# Best trial fold 1:
#   Value: 0.813887357711792
#   Number: 96
#   Params:
#     lr: 0.001727812188052317
#     weight_decay: 0.0021443244631774877
#     output_dropout: 0.12653500678301954

# Best trial fold 0:
#   Value: 0.646236777305603
#   Number: 68
#   Params:
#     lr: 0.001641662889508633
#     weight_decay: 0.0018992876846974842
#     output_dropout: 0.324013024009401