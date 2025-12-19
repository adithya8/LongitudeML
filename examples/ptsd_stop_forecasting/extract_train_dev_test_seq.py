from datasets import load_from_disk
import pandas as pd


if __name__ == '__main__':
    dataset_dev = load_from_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_roberta_laL23rpca64_wtcSubscalesNormalized_merged_PCL_1_days_ahead_max60days_v6_40combined_devset_oots')
    dataset_test = load_from_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_roberta_laL23rpca64_wtcSubscalesNormalized_merged_PCL_1_days_ahead_max90days_v6_60combined_5fold_oots')
    
    dataset_dev_train_seq = dataset_dev.filter(lambda x: x['folds'] != 0)
    dev_train_data = dataset_dev_train_seq.select_columns(['seq_id', 'time_ids']).to_list()
    dataset_dev_test_seq = dataset_dev.filter(lambda x: x['folds'] == 0)
    dev_test_data = dataset_dev_test_seq.select_columns(['seq_id', 'time_ids']).to_list()
    
    dataset_test_train_seq = dataset_test.filter(lambda x: x['folds'] != 4)
    test_train_data = dataset_test_train_seq.select_columns(['seq_id', 'time_ids']).to_list()
    dataset_test_test_seq = dataset_test.filter(lambda x: x['folds'] == 4)
    test_test_data = dataset_test_test_seq.select_columns(['seq_id', 'time_ids']).to_list()
    
    def format_to_df(data):
        df = pd.DataFrame(columns=['seq_id', 'time_ids'])
        for d in data:
            num_time_ids = len(d['time_ids'])
            temp_df = pd.DataFrame(zip([d['seq_id']]*num_time_ids, d['time_ids']), columns=['seq_id', 'time_ids'])
            temp_df['user_day_id'] = temp_df['seq_id'].astype(str) + '_' + temp_df['time_ids'].astype(str)
            df = pd.concat([df, temp_df], axis=0, ignore_index=True)
        return df
    
    dev_train_df = format_to_df(dev_train_data)
    dev_test_df = format_to_df(dev_test_data)
    test_train_df = format_to_df(test_train_data)
    test_test_df = format_to_df(test_test_data)
    
    dev_train_df.loc[:, 'is_dev_trainset'] = int(1)
    dev_test_df.loc[:, 'is_dev_trainset'] = int(0)
    dev_train_df.loc[:, 'is_dev_testset'] = int(0)
    dev_test_df.loc[:, 'is_dev_testset'] = int(1)
    dev_train_df.loc[:, 'is_dev_ooss'] = int(0)
    dev_test_df.loc[:, 'is_dev_ooss'] = int(1)
    dev_train_df.loc[dev_train_df.time_ids < 40, 'is_dev_oost'] = int(0)
    dev_train_df.loc[dev_train_df.time_ids >= 40, 'is_dev_oost'] = int(1)
    dev_test_df.loc[dev_test_df.time_ids < 40, 'is_dev_oost'] = int(0)
    dev_test_df.loc[dev_test_df.time_ids >= 40, 'is_dev_oost'] = int(1)
    
    # Concat dev data
    dev_df = pd.concat([dev_train_df, dev_test_df], axis=0, ignore_index=True)

    test_train_df.loc[:, 'is_test_trainset'] = int(1)
    test_test_df.loc[:, 'is_test_trainset'] = int(0)
    test_train_df.loc[:, 'is_test_testset'] = int(0)
    test_test_df.loc[:, 'is_test_testset'] = int(1)
    test_train_df.loc[:, 'is_test_ooss'] = int(0)
    test_test_df.loc[:, 'is_test_ooss'] = int(1)
    test_train_df.loc[test_train_df.time_ids < 60, 'is_test_oost'] = int(0)
    test_train_df.loc[test_train_df.time_ids >= 60, 'is_test_oost'] = int(1)
    test_test_df.loc[test_test_df.time_ids < 60, 'is_test_oost'] = int(0)
    test_test_df.loc[test_test_df.time_ids >= 60, 'is_test_oost'] = int(1)
        
    # Concat test data
    test_df = pd.concat([test_train_df, test_test_df], axis=0, ignore_index=True)
    
    full_df = pd.merge(dev_df, test_df, on=['seq_id', 'time_ids', 'user_day_id'], how='right')
    full_df = full_df.fillna('NULL')
    
    # # Set datatype for the last 8 columns
    # full_df.loc[:, 'is_dev_trainset'] = full_df['is_dev_trainset'].astype(int)
    # full_df.loc[:, 'is_dev_testset'] = full_df['is_dev_testset'].astype(int)
    # full_df.loc[:, 'is_dev_ooss'] = full_df['is_dev_ooss'].astype(int)
    # full_df.loc[:, 'is_dev_oost'] = full_df['is_dev_oost'].astype(int)
    # full_df.loc[:, 'is_test_trainset'] = full_df['is_test_trainset'].astype(int)
    # full_df.loc[:, 'is_test_testset'] = full_df['is_test_testset'].astype(int)
    # full_df.loc[:, 'is_test_ooss'] = full_df['is_test_ooss'].astype(int)
    # full_df.loc[:, 'is_test_oost'] = full_df['is_test_oost'].astype(int)
    
    full_df.to_csv('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_roberta_laL23rpca64_wtcSubscalesNormalized_merged_PCL_1_days_ahead_max60days_v6_40combined_devset_oots_train_dev_test_seq.csv', index=False)