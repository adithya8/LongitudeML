python run_drinking_ans.py --data_file '/data/avirinchipur/EMI/datadicts/drinking_ans_avg.pkl' --do_train \
       --train_batch_size 32 --val_batch_size 32 --num_workers 4 --epochs 10 --lr 1e-4 \
       --predict_last_valid_timestep --output_dir '/data/avirinchipur/EMI/outputs/DS4UD/drinking_ans_avg' \
       --workspace EMI --project_name DS4UD --experiment_name drinking_ans_avg_1strun