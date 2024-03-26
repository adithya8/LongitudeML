import os
import argparse
import pickle
from utils import add_to_path
from copy import deepcopy

add_to_path(__file__)
from src import DLATKDataGetter

# feat$roberta_ba_meL11con$ds4ud_msgs_words_day_v9$message_id

base_dict = dict(db="EMI", msg_table="ds4ud_msgs_words_day_v9", feature_tables=["feat$dr_rpca_64_fb20_robaL11$ds4ud_msgs_words_day_v9$message_id", "feat$today_drinks$ds4ud_msgs_words_day_v9$message_id"],
                 outcome_table="ds4ud_wave_drinks_words_v9", outcome_field="", timeid_field="day_id", correl_field="usryrwv_id")

INPUT_DICT = {
    "drinks_ans_avg": deepcopy(base_dict)
}

for outcome in INPUT_DICT:
    INPUT_DICT[outcome]["outcome_field"] = outcome


def parse_args():
    parser = argparse.ArgumentParser(description='Get data from DLATK from DS4UD tables')
    parser.add_argument('--outcome_field', type=str, default='drinking_ans_avg', help='Outcome field', choices=INPUT_DICT.keys())
    parser.add_argument('--save_filepath', type=str, default='/data/avirinchipur/EMI/datadicts/default.pkl', help='Output file path to save the data dictionary')
    parser.add_argument('--val_ratio', type=float, default=0.0, help='Validation ratio')
    parser.add_argument('--val_fold_column', type=str, default=None, help='Column to use for validation fold')
    parser.add_argument('--test_ratio', type=float, default=0.0, help='Test ratio')
    parser.add_argument('--test_fold_column', type=str, default=None, help='Column to use for test fold')
    parser.add_argument('--num_folds', type=int, default=0, help='Number of folds for cross validation')
    parser.add_argument('--fold_column', type=str, default=None, help='Column to use for cross validation folds')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dlatk_dataGetter = DLATKDataGetter(**INPUT_DICT[args.outcome_field])
    dataDict, longtype_encoder = dlatk_dataGetter.combine_features_and_outcomes()
    dataDict = dlatk_dataGetter.clamp_sequence_length(dataDict, min_seq_len=3, max_seq_len=14, retain="last")

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