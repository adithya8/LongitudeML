python run_PCL.py --data_file '/data/avirinchipur/EMI/datadicts/ptsd_stop_PCL_avg_roba128_rangel40u80_5folds.pkl' --val_folds 0 \
       --do_hparam_tune --n_trials 100 \
       --train_batch_size 8 --val_batch_size 16 --num_workers 4 \
       --dropout 0.05 --output_dropout 0.1 --num_layer 1 --min_epochs 15 --epochs 100 --lr 5e-5 --weight_decay 1e-3 \
       --input_size 128 --hidden_size 64 \
       --early_stopping_patience 10 --early_stopping_min_delta 0.0 \
       --output_dir '/data/avirinchipur/EMI/outputs/ptsd_stop/PCL_avg_roba128' \
       --workspace EMI --project_name ptsd_stop_PCL_avg_folds0_v1 --experiment_name hp_roba128_1Lgru_ln


# Best trial from v2 fold 4:
#   Value: 0.5786484479904175
#   Number: 88
#   Params:
#     lr: 0.007771064541932739
#     weight_decay: 1.3118821131069e-05
#     output_dropout: 0.0006284610401790288
#     hidden_size: 64

# Best trial from v2 fold 3:
#   Value: 0.25505971908569336
#   Number: 34
#   Params:
#     lr: 0.003059399748257763
#     weight_decay: 1.852957919035083e-05
#     output_dropout: 0.4730628630266869

# Best trial from v2 fold 2:
#  Value: 0.5699118971824646
#   Number: 59
#   Params:
#     lr: 0.0021558725283466786
#     weight_decay: 1.4968399579055085e-05
#     output_dropout: 0.4500345362695106

# Best trial fold 1:
#   Value: 0.48556965589523315
#   Number: 96
#   Params:
#     lr: 0.001198368880341542
#     weight_decay: 0.0038192782727480864
#     output_dropout: 0.3579773486707952

# Best trial fold 0:
#   Value: 0.26235103607177734
#   Number: 97
#   Params:
#     lr: 0.000485544384946633
#     weight_decay: 0.0004987033039464467
#     output_dropout: 0.32572686926645467
