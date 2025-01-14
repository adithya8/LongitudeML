python run_drinking_ans.py --data_file '/data/avirinchipur/EMI/datadicts//ds4ud_drinks_ans_avg_words_roba768_todaydrinks_rangel3u14_10folds.pkl' --do_train \
       --train_batch_size 64 --val_batch_size 64 --num_workers 4 --num_layer 1 --epochs 200 --lr 5e-5  \
       --output_dir '/data/avirinchipur/EMI/outputs/DS4UD/drinking_ans_avg' --weight_decay 1e-1 \
       --workspace EMI --project_name DS4UD --experiment_name dummy_1
