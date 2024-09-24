import os
import argparse
import pickle
from utils import add_to_path
from copy import deepcopy

add_to_path(__file__)
from src import DLATKDataGetter

# Datadict location on Hercules:
# /data/avirinchipur/EMI/datadicts/

def parse_args():
    parser = argparse.ArgumentParser(description='Get data from DLATK from PTSD STOP tables')
    parser.add_argument('--outcome_field', type=str, default='PCL_avg', help='Outcome field', choices=INPUT_DICT.keys())
    parser.add_argument('--save_filepath', type=str, required=True, help='Output file path to save the data dictionary')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='Validation ratio')
    parser.add_argument('--val_fold_column', type=str, default=None, help='Column to use for validation fold')
    parser.add_argument('--test_ratio', type=float, default=0.0, help='Test ratio')
    parser.add_argument('--test_fold_column', type=str, default=None, help='Column to use for test fold')
    parser.add_argument('--num_folds', type=int, default=0, help='Number of folds for cross validation')
    parser.add_argument('--fold_column', type=str, default=None, help='Column to use for cross validation folds')
    return parser.parse_args()


base_dict = dict(db="ptsd_stop", msg_table="whisper_transcripts_v1", feature_tables=["feat$dr_rpca_128_fb20$whisper_transcripts_v1$user_day_id"],
                 outcome_table="outcomes_PCL_forecast_v2", outcome_field="", timeid_field="day_id", correl_field="user_id", messageid_field="user_day_id")

INPUT_DICT = {
    "today_PCL": deepcopy(base_dict),
    "PCL_1_day_ahead": deepcopy(base_dict),
    "PCL_2_days_ahead": deepcopy(base_dict),
    "PCL_4_days_ahead": deepcopy(base_dict),
    "PCL_7_days_ahead": deepcopy(base_dict),
}


for outcome in INPUT_DICT:
    INPUT_DICT[outcome]["outcome_field"] = outcome


if __name__ == '__main__':
    args = parse_args()
    dlatk_dataGetter = DLATKDataGetter(**INPUT_DICT[args.outcome_field])
    
    # NOTE: This will only work for user level outcome. get_outcomes() fetches outcomes only at correl_field level
    dataDict, longtype_encoder = dlatk_dataGetter.combine_features_and_outcomes(outcomes_correl_field=None if 'avg' in args.outcome_field 
                                                                                else 'user_day_id')
    dataDict = dlatk_dataGetter.clamp_sequence_length(dataDict, min_seq_len=40, max_seq_len=80, retain="first")
    
    if args.test_fold_column is not None or args.val_fold_column is not None:
        dataDict = dlatk_dataGetter.train_test_split(dataset_dict=dataDict, longtype_encoder=longtype_encoder, \
                                                            test_fold_column=args.test_fold_column, val_fold_column=args.val_fold_column)

    if args.fold_column is not None:
        dataDict = dlatk_dataGetter.n_fold_split(dataDict, longtype_encoder, fold_column=args.fold_column)

    if os.path.exists(args.save_filepath):
        print(f"File {args.save_filepath} already exists. Overwriting...")
    with open(args.save_filepath, 'wb') as f:
        pickle.dump(dataDict, f)
    encoder_filepath = args.save_filepath.replace('.pkl', '_id_encoder.pkl')
    with open(encoder_filepath, 'wb') as f:
        pickle.dump(longtype_encoder, f)
        
    print (f"Saved data dictionary to {args.save_filepath}")
    print (f"Saved encoder to {encoder_filepath}")