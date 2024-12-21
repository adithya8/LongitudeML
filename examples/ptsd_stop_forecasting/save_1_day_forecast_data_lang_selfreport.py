"""
    This script is responsible for reading the features, outcomes tables from MySQL database 
    and creating a dataset suitable for forecasting PCL scores 1, 3, 7 days ahead.
"""
from copy import deepcopy
from typing import List, Union
from utils import add_to_path
add_to_path(__file__)

from numpy import zeros_like as np_zeros_like
import numpy as np
from datasets import Dataset

from src import DLATKDataGetter
from src import get_dataset

def read_dlatk_data(base_dict:dict):
    dlatk_data_getter = DLATKDataGetter(**base_dict)
    long_embs_data = dlatk_data_getter.get_long_features()
    long_outcomes_data = dlatk_data_getter.get_long_outcomes()
    long_embs, long_outcomes = dlatk_data_getter.intersect_seqids(deepcopy(long_embs_data), deepcopy(long_outcomes_data))
    
    return long_embs, long_outcomes
 
 
def create_embs_mask(instance):
    """
        Creates a mask for Embeddings based on the following conditions:
        1. Infills missing value with the previous known embeddings. 
        2. If embeddings are present for a time_id, set mask = 0. Else set it to 1. 
        Objective of mask is to determine if an element needs to be masked (i.e., when mask = 1). Hence only the missing time_ids are masked.
    """
    
    sorted_time_ids = sorted(instance['time_ids'])
    missing_time_ids = set(range(sorted_time_ids[0], sorted_time_ids[-1]+1)) - set(sorted_time_ids)
    min_time_id = sorted_time_ids[0]
    infill_mask = [0]*len(instance['time_ids'])
    for time_id in sorted(missing_time_ids):
        infill_mask.insert(time_id-min_time_id, 1) # inserting 1 as mask at the missing time_id 
        instance['time_ids'].insert(time_id-min_time_id, time_id)
        # instance['embeddings'][0].insert(time_id-min_time_id, [0]*len(instance['embeddings'][0][time_id-min_time_id-1])) # 0 infilling is not better than copying the previous vector
        instance['embeddings'].insert(time_id-min_time_id, instance['embeddings'][time_id-min_time_id-1])
        instance['num_tokens'].insert(time_id-min_time_id, None)
    instance['mask'] = infill_mask
    
    return instance


def create_outcomes_mask(instance, outcomes_names:List[str]):
    """
        Creates a mask for outcomes based on the following conditions:
        1. Infills None and missing values with 0.
        2. If outcome is present for a time_id, set mask = 1. Else set it to 0.
        Outcome mask is used to determine whether loss is calculated for a time_id or not. Hence the missing time_ids are masked with 0.
    """
    num_outcomes = len(outcomes_names)
    sorted_time_ids = sorted(instance['time_ids'])
    # missing_time_ids = set(range(sorted_time_ids[0], sorted_time_ids[-1]+1)) - set(sorted_time_ids)
    missing_time_ids = set(range(0, sorted_time_ids[-1]+1)) - set(sorted_time_ids)
    min_time_id = sorted_time_ids[0]
    infill_mask = []  
    # for time_id in range(sorted_time_ids[0], sorted_time_ids[-1]+1):
    for time_id in range(0, sorted_time_ids[-1]+1):
        if time_id in missing_time_ids:
            infill_mask.insert(time_id-min_time_id, [0]*num_outcomes) # inserting 0 as mask at the missing time_id
            instance['outcomes'].insert(time_id-min_time_id, [0]*num_outcomes)
            instance['time_ids'].insert(time_id-min_time_id, time_id)
        else:
            temp_infill_mask = [1]*num_outcomes
            for idx in range(num_outcomes):
                if instance['outcomes'][time_id-min_time_id][idx] is None:
                    instance['outcomes'][time_id-min_time_id][idx] = 0
                    temp_infill_mask[idx] = 0
            infill_mask.insert(time_id-min_time_id, temp_infill_mask)                
    instance['outcomes_mask'] = infill_mask
    
    return instance


def merge_features_outcomes(long_outcomes:Dataset, long_features:Dataset):
    """
        Merges the features and outcomes datasets.
        1. Takes the min of time_ids between features and outcomes, max of time_ids between features and outcomes.
        2. Merges the datasets based on mutual time_ids, infills the missing time_ids in respective modalities with appropriate values.
    """
    
    # Find the name of the embeddings features in the long_features dataset
    embs_names = [key for key in long_features.features.keys() if key.startswith('embeddings')]
    masks_names = [key for key in long_features.features.keys() if key.startswith('mask')]
    
    # Create merged_dataset dictionary with the same keys as long_features in addition to outcomes and outcomes_mask
    merged_dataset = {key: [] for key in long_features.features.keys()}
    merged_dataset['outcomes'] = []
    merged_dataset['outcomes_mask'] = []
    
    def unionize_time_ids(outcomes_instance, features_instance):
        """
            Returns the union of time_ids between lang and outcomes instances.
        """
        merged_instance = {"seq_id": features_instance['seq_id'], "time_ids": [], "outcomes": [], "outcomes_mask": []}
        merged_instance.update({key: [] for key in embs_names})
        merged_instance.update({key: [] for key in masks_names})
        outcomes_time_ids = set(outcomes_instance['time_ids'])
        features_time_ids = set(features_instance['time_ids'])
        union_time_ids = features_time_ids.union(outcomes_time_ids)
        infill_outcomes = [0]*len(outcomes_instance['outcomes'][0])
        infill_outcomes_mask = [0]*len(outcomes_instance['outcomes_mask'][0])
        infill_features = dict([(emb_name, np_zeros_like(features_instance[emb_name][0]).tolist()) for emb_name in embs_names])
        feature_timeids_idx = dict(zip(features_instance['time_ids'], range(len(features_instance['time_ids']))))
        outcomes_timeids_idx = dict(zip(outcomes_instance['time_ids'], range(len(outcomes_instance['time_ids']))))
        for time_id in range(min(union_time_ids), max(union_time_ids)+1):
            if time_id not in features_time_ids:
                for emb_name in embs_names: merged_instance[emb_name].append(infill_features[emb_name])
                for mask_name in masks_names: merged_instance[mask_name].append(1)
            else:
                feature_time_idx = feature_timeids_idx[time_id]
                for emb_name in embs_names: merged_instance[emb_name].append(features_instance[emb_name][feature_time_idx])
                for mask_name in masks_names: merged_instance[mask_name].append(features_instance[mask_name][feature_time_idx])
                infill_features = dict([(emb_name, features_instance[emb_name][feature_time_idx]) for emb_name in embs_names])
            if time_id not in outcomes_time_ids:
                merged_instance["outcomes"].append(infill_outcomes)
                merged_instance["outcomes_mask"].append(infill_outcomes_mask)
            else:
                outcomes_time_idx = outcomes_timeids_idx[time_id]
                merged_instance["outcomes"].append(outcomes_instance["outcomes"][outcomes_time_idx])
                merged_instance["outcomes_mask"].append(outcomes_instance["outcomes_mask"][outcomes_time_idx])        
        merged_instance["time_ids"] = list(range(min(union_time_ids), max(union_time_ids)+1))
        return merged_instance
    
    mutual_seq_ids = set(long_features['seq_id']) & set(long_outcomes['seq_id'])
    for seq_id in mutual_seq_ids:
        features_seqid_idx = long_features['seq_id'].index(seq_id)
        features_instance = long_features[features_seqid_idx]
        outcomes_seqid_idx = long_outcomes['seq_id'].index(seq_id)
        outcomes_instance = long_outcomes[outcomes_seqid_idx]
        merged_instance = unionize_time_ids(outcomes_instance, features_instance)
        for key in merged_instance:
            merged_dataset[key].append(merged_instance[key])
        
    return get_dataset(merged_dataset)


def concatenate_features(feature_dataset1, feature_dataset2):
    """
        Concatenates the features from two dictionaries.
        1. Takes the min of time_ids between feature1 and feature2, max of time_ids between feature1 and feature2.
        2. Merges the datasets based on mutual time_ids, infills the missing time_ids with the last previous valid feature value. Mask is set to 1when both features are missing. 
    """
    concatenated_features = {"seq_id": [], "embeddings": [], "num_tokens": [], "time_ids": [], "mask": []}
    def unionize_time_ids(feat1_instance, feat2_instance):
        """
            Returns the union of time_ids between lang and outcomes instances.
        """
        concat_instance = {"seq_id": feat1_instance['seq_id'], "embeddings": [], "num_tokens": [], "lang_time_ids": [], "mask": []}
        feat1_time_ids = set(feat1_instance['time_ids'])
        feat2_time_ids = set(feat2_instance['time_ids']) 
        union_time_ids = feat1_time_ids.union(feat2_time_ids)
        min_feat1_time_id, min_feat2_time_id = min(feat1_time_ids), min(feat2_time_ids)
        for time_id in range(min(union_time_ids), max(union_time_ids)+1):
            if time_id in feat1_time_ids and time_id in feat2_time_ids:
                # concatenate the embeddings and take the max(num_tokens) for the time_id
                concat_embeddings = feat1_instance['embeddings'][feat1_instance['time_ids'].index(time_id)] + feat2_instance['embeddings'][feat2_instance['time_ids'].index(time_id)]
                concat_num_tokens = feat2_instance['num_tokens'][feat2_instance['time_ids'].index(time_id)]
                concat_instance["embeddings"].append(concat_embeddings)
                concat_instance["num_tokens"].append(concat_num_tokens)
                concat_instance["lang_time_ids"].append(time_id)
                infill_embs_mask = (feat1_instance['mask'][feat1_instance['time_ids'].index(time_id)]) \
                                    & (feat2_instance['mask'][feat2_instance['time_ids'].index(time_id)])
                concat_instance["mask"].append(infill_embs_mask)
                prev_feat1_embs = feat1_instance['embeddings'][feat1_instance['time_ids'].index(time_id)]
                prev_feat2_embs = feat2_instance['embeddings'][feat2_instance['time_ids'].index(time_id)]
                
            elif time_id in feat1_time_ids:
                feat2_embs_infill = prev_feat2_embs if time_id > min_feat2_time_id else np_zeros_like(feat2_instance['embeddings'][0]).tolist()
                concat_embeddings = feat1_instance['embeddings'][feat1_instance['time_ids'].index(time_id)] + feat2_embs_infill
                concat_num_tokens = feat1_instance['num_tokens'][feat1_instance['time_ids'].index(time_id)]
                concat_instance["embeddings"].append(concat_embeddings)
                concat_instance["num_tokens"].append(concat_num_tokens)
                concat_instance["lang_time_ids"].append(time_id)
                infill_embs_mask = feat1_instance['mask'][feat1_instance['time_ids'].index(time_id)] & 1
                concat_instance["mask"].append(infill_embs_mask)
                concat_instance["mask"].append(0)
                prev_feat1_embs = feat1_instance['embeddings'][feat1_instance['time_ids'].index(time_id)]
                prev_feat2_embs = feat2_embs_infill
            
            elif time_id in feat2_time_ids:
                feat1_embs_infill = prev_feat1_embs if time_id > min_feat1_time_id else np_zeros_like(feat1_instance['embeddings'][0]).tolist()
                concat_embeddings = feat1_embs_infill + feat2_instance['embeddings'][feat2_instance['time_ids'].index(time_id)]
                concat_num_tokens = feat2_instance['num_tokens'][feat2_instance['time_ids'].index(time_id)]
                concat_instance["embeddings"].append(concat_embeddings)
                concat_instance["num_tokens"].append(concat_num_tokens)
                concat_instance["lang_time_ids"].append(time_id)
                infill_embs_mask = feat1_instance['mask'][feat2_instance['time_ids'].index(time_id)] & 1
                concat_instance["mask"].append(infill_embs_mask)
                prev_feat1_embs = feat1_embs_infill
                prev_feat2_embs = feat2_instance['embeddings'][feat2_instance['time_ids'].index(time_id)]
                
            else:
                feat1_embs_infill = prev_feat1_embs if time_id > min_feat1_time_id else np_zeros_like(feat1_instance['embeddings'][0]).tolist()
                feat2_embs_infill = prev_feat2_embs if time_id > min_feat2_time_id else np_zeros_like(feat2_instance['embeddings'][0]).tolist()
                concat_embeddings = feat1_embs_infill + feat2_embs_infill
                concat_num_tokens = 0
                concat_instance["embeddings"].append(concat_embeddings)
                concat_instance["num_tokens"].append(concat_num_tokens)
                concat_instance["lang_time_ids"].append(time_id)
                concat_instance["mask"].append(1) # If both time_ids are missing, then mask is set 1 
                prev_feat1_embs = feat1_embs_infill
                prev_feat2_embs = feat2_embs_infill
        return concat_instance        
        
    mutual_seq_ids = set(feature_dataset1['seq_id']) & set(feature_dataset2['seq_id'])
    for seq_id in mutual_seq_ids:
        feat1_seqid_idx = feature_dataset1['seq_id'].index(seq_id)
        feat1_instance = feature_dataset1[feat1_seqid_idx]
        feat2_seqid_idx = feature_dataset2['seq_id'].index(seq_id)
        feat2_instance = feature_dataset2[feat2_seqid_idx]
        concat_instance = unionize_time_ids(feat1_instance, feat2_instance)
        concatenated_features["seq_id"].append(seq_id)
        concatenated_features["embeddings"].append(concat_instance["embeddings"])
        concatenated_features["num_tokens"].append(concat_instance["num_tokens"])
        concatenated_features["time_ids"].append(concat_instance["lang_time_ids"])
        concatenated_features["mask"].append(concat_instance["mask"])

    return get_dataset(concatenated_features)


def merge_features(feature_datasets:List[Dataset], feature_suffixes:List[str]=None):
    """
        Merges the features from two dictionaries. In this case, the feature vectors are merged into one feature dictionary, while keeping the features and their masks separated.\n
        Two sets of Features which come from either of get_long_features() or concatenate_features() can be merged using this function.\n
        The merged features contain two different sets of features and their masks.\n
        Usage:
            `merged_features = merge_features([long_sr_features, long_lang_features], feature_suffixes=['_sr', '_lang'])``
    """
    
    assert len(feature_datasets) >= 2, "At least two feature datasets are required for merging."
    assert feature_suffixes is None or len(feature_suffixes) == len(feature_datasets), "Feature suffixes should be equal to the number of feature datasets."
    # assert that non None feature_suffixes are unique
    if feature_suffixes is not None:
        feature_suffixes_noNone = [suffix for suffix in feature_suffixes if suffix is not None]
        assert len(set(feature_suffixes_noNone)) == len(feature_suffixes_noNone), "Feature suffixes should be unique."
        
    def get_embs_name_suffix(feature_suffixes):
        feature_names = [[key for key in feature_dataset.features.keys() if key.startswith('embeddings')] for feature_dataset in feature_datasets]  
        all_feature_suffixes = set()
        if feature_suffixes is None: 
            feature_suffixes = []
            for idx, embs_name in enumerate(feature_names):
                temp_suffix = []
                for jdx, name in enumerate(embs_name):
                    if name not in all_feature_suffixes: 
                        temp_suffix.append('')
                        all_feature_suffixes.add(name)
                    else:
                        temp_suffix.append('_f{}_{}'.format(idx, jdx))
                        all_feature_suffixes.add('{}_f{}_{}'.format(name, idx, jdx))
                feature_suffixes.append(temp_suffix)
        else:
            temp_suffixes = []  
            all_feature_suffixes = set()
            for idx, embs_name in enumerate(feature_names):                
                temp_suffix = []
                for jdx, name in enumerate(embs_name):
                    if feature_suffixes[idx] is None: feature_suffixes[idx] = ''
                    if f'{name}{feature_suffixes[idx]}' not in all_feature_suffixes:
                        temp_suffix.append(feature_suffixes[idx])
                        all_feature_suffixes.add(f'{name}{feature_suffixes[idx]}')
                    else:
                        temp_suffix.append(f'_f{idx}_{jdx}')
                        all_feature_suffixes.add(f'{name}_f{idx}_{jdx}')
                temp_suffixes.append(temp_suffix)
            feature_suffixes = temp_suffixes
        
        return feature_names, feature_suffixes        
    
    feature_names, feature_suffixes = get_embs_name_suffix(feature_suffixes)
    merged_features = {"seq_id": [], "time_ids": []}
    for suffixes in feature_suffixes:
        for suffix in suffixes:
            merged_features["embeddings{}".format(suffix)] = []
            merged_features["mask{}".format(suffix)] = []
            
    mutual_seq_ids = set(feature_datasets[0]['seq_id'])
    for feature_dataset in feature_datasets[1:]:
        mutual_seq_ids = mutual_seq_ids & set(feature_dataset['seq_id'])

    def merge_feats(feature_instances):
        merged_instance = {"seq_id": feature_instances[0]['seq_id'], "time_ids": []}
        for suffixes in feature_suffixes:
            for suffix in suffixes:
                merged_instance["embeddings{}".format(suffix)] = []
                merged_instance["mask{}".format(suffix)] = []
        
        feature_time_ids = [set(feature_instance['time_ids']) for feature_instance in feature_instances]
        union_time_ids = sorted(feature_time_ids[0].union(*feature_time_ids[1:]))
        infill_emb_feats = dict([(emb_name, np_zeros_like(feature_instances[idx][emb_name][0]).tolist()) for idx, embs_names in enumerate(feature_names) for jdx, emb_name in enumerate(embs_names)])
        for time_id in union_time_ids:
            for idx, feature_instance in enumerate(feature_instances):
                if time_id in feature_instance['time_ids']:
                    feature_time_idx = feature_instance['time_ids'].index(time_id)
                    for suffix, feature_name in zip(feature_suffixes[idx], feature_names[idx]):
                        merged_instance["{}{}".format(feature_name, suffix)].append(feature_instance[feature_name][feature_time_idx])
                        merged_instance["mask{}".format(suffix)].append(feature_instance['mask'][feature_time_idx])
                        infill_emb_feats[feature_name] = feature_instance[feature_name][feature_time_idx]
                else:
                    for suffix in feature_suffixes[idx]:
                        merged_instance["{}{}".format(feature_name, suffix)].append(infill_emb_feats[feature_names[idx][0]])
                        merged_instance["mask{}".format(suffix)].append(1)
            merged_instance["time_ids"].append(time_id)
        return merged_instance
                    
    for seq_id in mutual_seq_ids:
        feature_seq_idxs = [feature_dataset['seq_id'].index(seq_id) for feature_dataset in feature_datasets]
        feature_instances = [feature_dataset[feature_seq_idx] for feature_dataset, feature_seq_idx in zip(feature_datasets, feature_seq_idxs)]
        merged_instance = merge_feats(feature_instances)
        for key in merged_instance:
            merged_features[key].append(merged_instance[key])
    
    return get_dataset(merged_features)        
        

if __name__ == '__main__':
    
    base_dict = dict(db="ptsd_stop", msg_table="whisper_transcripts_v4", 
            feature_tables=["feat$PCL_sr$outcomes_v4$user_day_id"],
            # feature_tables=["feat$dr_rpca_32_roba_meL11$whisper_transcripts_v3$user_day_id"],
            outcome_table="outcomes_v4_PCL_1_day_ahead", outcome_fields=["PCL_1_day_ahead"], timeid_field="day_id", 
            correl_field="user_id", messageid_field="user_day_id")
    
    long_sr_features, long_outcomes = read_dlatk_data(base_dict=base_dict)
    
    if 'embeddings_names' in long_sr_features:
        embeddings_names = long_sr_features.pop('embeddings_names')
        long_sr_features = get_dataset(long_sr_features)
    
    if 'outcomes_names' in long_outcomes:
        outcomes_names = long_outcomes.pop('outcomes_names')
        long_outcomes = get_dataset(long_outcomes)

    print ("Long SR features and Long Outcomes read.")
    # NOTE: This data is not already split for train/ val/ test. Perform filtering and train-val split here
    
    def minmax_normalization(instance):
        for idx, embs in enumerate(instance['embeddings']):
            embs_np = np.array(embs)
            instance['embeddings'][idx] = ((embs_np - 1.0) / (5.0 - 1.0)).tolist()
        return instance

    long_sr_features = long_sr_features.map(minmax_normalization)

    # CHECK for user 3586 before and after creating mask
    long_sr_features = long_sr_features.map(create_embs_mask)
    print ("SR mask pattern created.")
    
    base_dict = dict(db="ptsd_stop", msg_table="whisper_transcripts_v4", 
                feature_tables=["feat$dr_rpca_64_roba_meL11$whisper_transcripts_v4$user_day_id"],
                # feature_tables=["feat$dr_rpca_32_roba_meL11$whisper_transcripts_v3$user_day_id"],
                outcome_table="outcomes_v4_PCL_1_day_ahead", outcome_fields=["PCL_1_day_ahead"], timeid_field="day_id", 
                correl_field="user_id", messageid_field="user_day_id")
    
    long_lang_features, _ = read_dlatk_data(base_dict=base_dict)
    
    if 'embeddings_names' in long_lang_features:
        embeddings_names = long_lang_features.pop('embeddings_names')
        long_lang_features = get_dataset(long_lang_features)
    
    print ("Long Lang features and Long Outcomes read.")
    # NOTE: This data is not already split for train/ val/ test. Perform filtering and train-val split here
    
    # CHECK for user 3586 before and after creating mask
    long_lang_features = long_lang_features.map(create_embs_mask)
    print ("Lang mask pattern created.")
    
    base_dict = dict(db="ptsd_stop", msg_table="whisper_transcripts_v4",
                feature_tables=["feat$cat_dd_hypLex_w$whisper_transcripts_v4$user_day_id$1gra"],
                outcome_table="outcomes_v4_PCL_1_day_ahead", outcome_fields=["PCL_1_day_ahead"], timeid_field="day_id",
                correl_field="user_id", messageid_field="user_day_id")
    
    long_hypLex_features, _ = read_dlatk_data(base_dict=base_dict)
    
    if 'embeddings_names' in long_hypLex_features:
        embeddings_names = long_hypLex_features.pop('embeddings_names')
        long_hypLex_features = get_dataset(long_hypLex_features)
        
    def get_normalized_embeddings(dataset):
        # Compute mean for each sequence across each dimension
        # Comute the mean and std dev for each dimension on the mean of the sequences
        # Normalize the embeddings using the mean and std dev
        def compute_mean(instance): return np.mean(instance['embeddings'], axis=0)
        means = []
        for idx in range(len(dataset['seq_id'])):
            instance = dataset[idx]
            means.append(compute_mean(instance))
        mean_means = np.mean(means, axis=0)
        stddev_means = np.std(means, axis=0)
        def normalize_embeddings(instance):
            instance['embeddings'] = ((np.array(instance['embeddings']) - mean_means) / stddev_means).tolist()
            return instance
        dataset = dataset.map(normalize_embeddings)
        return dataset
    
    long_hypLex_features = get_normalized_embeddings(long_hypLex_features)
    print ("Long HypLex features and Long Outcomes read.")
    
    long_hypLex_features = long_hypLex_features.map(create_embs_mask)
    print ("HypLex mask pattern created.")
    
    base_dict = dict(db="ptsd_stop", msg_table="whisper_transcripts_v4",
                    feature_tables=["feat$p_ridg_Pcl_wtc$whisper_transcripts_v4$user_day_id"],
                    outcome_table="outcomes_v4_PCL_1_day_ahead", outcome_fields=["PCL_1_day_ahead"], timeid_field="day_id",
                    correl_field="user_id", messageid_field="user_day_id")
    
    long_wtc_pclsubscales, _ = read_dlatk_data(base_dict=base_dict)
    
    if 'embeddings_names' in long_wtc_pclsubscales:
        embeddings_names = long_wtc_pclsubscales.pop('embeddings_names')
        long_wtc_pclsubscales = get_dataset(long_wtc_pclsubscales)
    
    long_wtc_pclsubscales = get_normalized_embeddings(long_wtc_pclsubscales)
    
    long_wtc_pclsubscales = long_wtc_pclsubscales.map(create_embs_mask)
    print ("WTC PCL subscales mask pattern created.")
    
    long_outcomes = long_outcomes.map(lambda x: create_outcomes_mask(x, outcomes_names))
    print ("Outcomes mask pattern created.")

    long_merged_features = merge_features([long_sr_features, long_lang_features, long_hypLex_features, long_wtc_pclsubscales], ['_subscales', '_lang', '_hypLex', '_wtcSubscales'])
    
    # long_comb_features = concatenate_features(long_sr_features, long_lang_features)
    
    merged_dataset = merge_features_outcomes(long_outcomes, long_merged_features)
    print ("Datasets merged.")
   
    # Truncate all sequences to 90 time_ids of records, ie., the maximum value of time_id should be 89
    def truncate_sequences(instance, max_time_id):
        time_ids = instance['time_ids']
        if time_ids[-1] > max_time_id:
            # find the first time_id that is greater than max_time_id. The data can have missing time_ids
            idx = next((i for i, time_id in enumerate(time_ids) if time_id > max_time_id), None)
            for key in instance:
                if isinstance(instance[key], list): instance[key] = instance[key][:idx]
        return instance
    
    merged_dataset = merged_dataset.map(lambda x: truncate_sequences(x, 59))
    
    # Filter to sequences with at least 40 time_ids with either language or outcomes
    # To do that, sum infill_embs_mask and infill_outcomes_mask to the dataset and filter if the number of 1s/2s in the mask is less than 40. 
    # Only consider the first outcome_index of infill_outcomes_mask
    def filter_sequences(instance):
        embs_mask = [1 if i==0 else 0 for i in instance['mask_lang']]
        outcomes_mask = [mask[0] for mask in instance['outcomes_mask']]
        summed_mask = [(l+o)>0 for l, o in zip(embs_mask, outcomes_mask)]
        return sum(summed_mask) >= 40
    
    merged_dataset = merged_dataset.filter(filter_sequences)
    
    # Split the dataset into train, val
    # Stratify based on mean the stddev of the first outcome.
    # Calculate mean and std dev for each seq_id over first outcome when the mask is 1
    # Then split it into n parts based on number of folds
    # Iterate of each part and randomly assign (0, n-1) to each seq_id
    # Store this value as fold_num in the dataset
    
    num_folds=5
    def compute_mean_std(instance):
        outcomes = [outcome[0] for outcome, mask in zip(instance['outcomes'], instance['outcomes_mask']) if mask[0]==1]
        mean_outcome = np.mean(outcomes)
        std_outcome = np.std(outcomes)
        instance['avg_outcome'] = mean_outcome
        instance['std_outcome'] = std_outcome
        return instance
    
    np.random.seed(42)
    def stratify_sequences(batch):
        batch["folds"] = np.random.permutation(len(batch["seq_id"])) #np.random.randint(0, num_folds, len(batch["seq_id"]))
        return batch
    
    merged_dataset = merged_dataset.map(compute_mean_std)
    merged_dataset = merged_dataset.sort('std_outcome').sort('avg_outcome')
    merged_dataset = merged_dataset.map(stratify_sequences, batched=True, batch_size=num_folds, remove_columns=['avg_outcome', 'std_outcome'])
    
    # TODO: Create a function that would add another key:value pair to this dataset object maintaining a list for each sequence indicating whether a time_id belongs to train or eval. 
    # The function definition: def create_train_eval_time_mask(instance, train_time_ids: List[int], eval_time_ids: List[int]=None) -> dict:
    # Key = "oots_mask"; This value will be 0 for train_time_ids and 1 for eval_time_ids.
    # Figure the max applicable time_id for each sequence.
    # Every time_id not in train_time_id is eval_time_id for each sequence. Apply assertion that train_time_ids and eval_time_ids are mutually exclusive, but together they should cover all time_ids for that sequence.
    
    
    # merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/todayPCL_selfreport_PCL_1_days_ahead_max90days_v3_40combined_5fold')
    # merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_roberta_base_L11_rpca64_combined_PCL_1_days_ahead_max90days_v4_40combined_5fold')
    # merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_noNULLs_roberta_base_L11_rpca64_merged_PCL_1_days_ahead_max90days_v4_40combined_5fold')
    # merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_noNULLs_roberta_base_L11_rpca64_hypLexNormalized_merged_PCL_1_days_ahead_max60days_v4_40combined_5fold')
    merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_noNULLs_roberta_base_L11_rpca64_hypLexNormalized_wtcSubscalesNormalized_merged_PCL_1_days_ahead_max60days_v4_40combined_5fold')