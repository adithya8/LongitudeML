import os
import argparse
import pickle
from utils import add_to_path

add_to_path(__file__)
from src import DLATKDataGetter

base_dict = dict(db="EMI", msg_table="wtc_calls_msgs_cleaned", feature_table="feat$roberta_ba_meL11con$wtc_calls_msgs_cleaned$message_id",
                 outcome_table="RIFT_EMA_cleaned_aggregated", outcome_field="", correl_field="user_id")

INPUT_DICT = {
    "PCL11_avg": base_dict,
    "PCL11_ans_avg": base_dict,
    "IDAS_dep_avg": base_dict,
    "IDAS_dep_ans_avg": base_dict
}

for outcome in INPUT_DICT:
    INPUT_DICT[outcome]["outcome_field"] = outcome

def parse_args():
    parser = argparse.ArgumentParser(description='Get data from DLATK from DS4UD tables')
    parser.add_argument('--outcome_field', type=str, default='drinking_ans_avg', help='Outcome field', choices=INPUT_DICT.keys())
    parser.add_argument('--save_filepath', type=str, help='Output file path to save the data dictionary')#, default='/data/avirinchipur/EMI/datadicts/default.pkl')
    parser.add_argument('--test_ratio', type=float, default=0.15, help='Test ratio')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dlatk_dataGetter = DLATKDataGetter(**INPUT_DICT[args.outcome_field])
    dataDict, longtype_encoder = dlatk_dataGetter.combine_features_and_outcomes()
    dataDict = dlatk_dataGetter.clamp_sequence_length(dataDict, min_seq_len=3, max_seq_len=15, retain="last")

    if args.test_ratio > 0: dataDict = dlatk_dataGetter.train_test_split(dataDict, test_ratio=args.test_ratio)
    
    if os.path.exists(args.save_filepath):
        print(f"File {args.save_filepath} already exists. Overwriting...")
    with open(args.save_filepath, 'wb') as f:
        pickle.dump(dataDict, f)
    encoder_filepath = args.save_filepath.replace('.pkl', '_id_encoder.pkl')
    with open(encoder_filepath, 'wb') as f:
        pickle.dump(longtype_encoder, f)