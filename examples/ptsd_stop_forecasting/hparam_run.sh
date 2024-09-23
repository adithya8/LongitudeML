
# Check if the number of arguments is 1 
if [ "$#" -ne 1 ]; then
    echo "Illegal number of parameters"
    echo "Usage: $0 <fold_num>"
    exit 1
fi

fold_num=$1
project_name='ptsd_stop_PCL_1d_ahead_folds'$fold_num'_v1'
echo "Fold number: $fold_num"
echo "Project name: $project_name"

python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_1day_ahead_roba128_rangel40u80_5folds.pkl' --val_folds $fold_num \
       --do_hparam_tune --n_trials 100 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --dropout 0.05 --output_dropout 0.1 --num_layer 1 --min_epochs 15 --epochs 100 --lr 5e-5 --weight_decay 1e-3 \
       --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_future_roba128' \
       --workspace EMI --project_name $project_name --experiment_name hp_roba128_1Lgru_ln


# Best trial from v2 fold 4:
#   Value: 0.534163773059845
#   Number: 45
#   Params:
#     lr: 0.0012851574398024769
#     weight_decay: 0.005378633494497839
#     output_dropout: 0.029737305773107816

# Best trial from v2 fold 3:
#   Value: 0.34239983558654785
#   Number: 87
#   Params:
#     lr: 0.008942278225458277
#     weight_decay: 9.769241485999559e-05
#     output_dropout: 0.05271797329143185

# Best trial from v2 fold 2:
#   Value: 0.6183879375457764
#   Number: 93
#   Params:
#     lr: 0.003649724813432039
#     weight_decay: 1.0665810289482508e-05
#     output_dropout: 0.4887072523599267

# Best trial fold 1:
# Value: 1.2141118049621582
#   Number: 33
#   Params:
#     lr: 0.000498075939384542
#     weight_decay: 1.7063350559026258e-05
#     output_dropout: 0.2781535779174923

# Best trial fold 0:
#   Value: 0.7581160068511963
#   Number: 55
#   Params:
#     lr: 0.0038440298490446697
#     weight_decay: 3.729612024794069e-05
#     output_dropout: 0.42685551979656755