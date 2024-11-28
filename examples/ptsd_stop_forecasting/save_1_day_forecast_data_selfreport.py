"""
    This script is responsible for reading the features, outcomes tables from MySQL database 
    and creating a dataset suitable for forecasting PCL scores 1, 3, 7 days ahead.
"""
from copy import deepcopy
from utils import add_to_path
add_to_path(__file__)

from numpy import zeros_like as np_zeros_like
import numpy as np

from src import DLATKDataGetter
from src import get_dataset

def read_dlatk_data():
    # base_dict = dict(db="ptsd_stop", msg_table="whisper_transcripts_v4", 
    #             feature_tables=["feat$today_PCL$outcomes_v3_PCL_forecast$user_day_id"],
    #             # feature_tables=["feat$dr_rpca_32_roba_meL11$whisper_transcripts_v3$user_day_id"],
    #             outcome_table="outcomes_v3_PCL_forecast", outcome_fields=["PCL_1_day_ahead"], timeid_field="day_id", 
    #             correl_field="user_id", messageid_field="user_day_id")
    
    base_dict = dict(db="ptsd_stop", msg_table="whisper_transcripts_v4", 
            feature_tables=["feat$PCL_sr$outcomes_v3$user_day_id"],
            # feature_tables=["feat$dr_rpca_32_roba_meL11$whisper_transcripts_v3$user_day_id"],
            outcome_table="outcomes_v3_PCL_1_day_ahead", outcome_fields=["PCL_1_day_ahead"], timeid_field="day_id", 
            correl_field="user_id", messageid_field="user_day_id")

    dlatk_data_getter = DLATKDataGetter(**base_dict)
    long_lang_data = dlatk_data_getter.get_long_lang_features()
    long_outcomes_data = dlatk_data_getter.get_long_outcomes()
    long_lang_features, long_outcomes = dlatk_data_getter.intersect_seqids(deepcopy(long_lang_data), deepcopy(long_outcomes_data))
    
    return long_lang_features, long_outcomes
 
 
def create_lang_mask(instance):
    """
        Creates a mask for language features based on the following conditions:
        1. Infills missing value with a the previous known language embedding. 
        2. If language is present for a time_id, set mask = 0. Else set it to 1. 
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
    instance['infill_lang_mask'] = infill_mask
    
    return instance


def create_outcomes_mask(instance):
    """
        Creates a mask for outcomes based on the following conditions:
        1. Infills None and missing values with 0.
        2. If outcome is present for a time_id, set mask = 1. Else set it to 0.
        Outcome mask is used to determine whether loss is calculated for a time_id or not. Hence the missing time_ids are masked with 0.
    """
    num_outcomes = len(outcomes_names)
    sorted_time_ids = sorted(instance['time_ids'])
    missing_time_ids = set(range(sorted_time_ids[0], sorted_time_ids[-1]+1)) - set(sorted_time_ids)
    min_time_id = sorted_time_ids[0]
    infill_mask = []  
    for time_id in range(sorted_time_ids[0], sorted_time_ids[-1]+1):
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
    instance['infill_outcomes_mask'] = infill_mask
    
    return instance


def merge_datasets(long_lang_features, long_outcomes):
    """
        Merges the language features and outcomes datasets.
        1. Takes the min of time_ids between lang and outcomes, max of time_ids between lang and outcomes.
        2. Merges the datasets based on mutual time_ids, infills the missing time_ids in respective modalities with appropriate values.
    """
    
    merged_dataset = {"seq_id": [], "embeddings": [], "num_tokens": [], "time_ids": [], "infill_lang_mask": [], "outcomes": [], "infill_outcomes_mask": []}
    def unionize_time_ids(lang_instance, outcomes_instance):
        """
            Returns the union of time_ids between lang and outcomes instances.
        """
        merged_instance = {"seq_id": lang_instance['seq_id'], "embeddings": [], "num_tokens": [], "lang_time_ids": [], "outcomes_time_ids": [], 
                           "infill_lang_mask": [], "outcomes": [], "infill_outcomes_mask": []}
        lang_time_ids = set(lang_instance['time_ids'])
        outcomes_time_ids = set(outcomes_instance['time_ids']) 
        union_time_ids = lang_time_ids.union(outcomes_time_ids)
        min_lang_time_id = min(lang_time_ids)
        for time_id in range(min(union_time_ids), max(union_time_ids)+1):
            if time_id in lang_time_ids:                
                merged_instance["embeddings"].append(lang_instance['embeddings'][lang_instance['time_ids'].index(time_id)])
                merged_instance["num_tokens"].append(lang_instance['num_tokens'][lang_instance['time_ids'].index(time_id)])
                merged_instance["lang_time_ids"].append(time_id)
                merged_instance["infill_lang_mask"].append(lang_instance['infill_lang_mask'][lang_instance['time_ids'].index(time_id)])
            else:
                infill_embedding = merged_instance["embeddings"][-1] if time_id > min_lang_time_id else np_zeros_like(lang_instance['embeddings'][0])
                merged_instance["embeddings"].append(infill_embedding)
                merged_instance["num_tokens"].append(None)
                merged_instance["lang_time_ids"].append(time_id)
                merged_instance["infill_lang_mask"].append(1) #Needs to be masked
            if time_id in outcomes_time_ids:
                merged_instance["outcomes"].append(outcomes_instance['outcomes'][outcomes_instance['time_ids'].index(time_id)])
                merged_instance["infill_outcomes_mask"].append(outcomes_instance['infill_outcomes_mask'][outcomes_instance['time_ids'].index(time_id)])
                merged_instance["outcomes_time_ids"].append(time_id)
            else:
                infill_outcome = [0]*len(outcomes_instance['outcomes'][0])
                merged_instance["outcomes"].append(infill_outcome)
                merged_instance["infill_outcomes_mask"].append([0]*len(outcomes_instance['outcomes'][0]))
                merged_instance["outcomes_time_ids"].append(time_id)
                
        if merged_instance["lang_time_ids"] != merged_instance["outcomes_time_ids"]:
            print ('------------------------------------')
            print ("Time_ids mismatch between lang and outcomes instances for seq_id ({}).".format(merged_instance["seq_id"]))
            print ("This might cause issues downstream")  
            print ('------------------------------------')          

        return merged_instance
    
    mutual_seq_ids = set(long_lang_features['seq_id']) & set(long_outcomes['seq_id'])
    for seq_id in mutual_seq_ids:
        lang_seqid_idx = long_lang_features['seq_id'].index(seq_id)
        lang_instance = long_lang_features[lang_seqid_idx]
        outcomes_seqid_idx = long_outcomes['seq_id'].index(seq_id)
        outcomes_instance = long_outcomes[outcomes_seqid_idx]
        merged_instance = unionize_time_ids(lang_instance, outcomes_instance)
        merged_dataset["seq_id"].append(seq_id)
        merged_dataset["embeddings"].append(merged_instance["embeddings"])
        merged_dataset["num_tokens"].append(merged_instance["num_tokens"])
        merged_dataset["time_ids"].append(merged_instance["lang_time_ids"])
        merged_dataset["infill_lang_mask"].append(merged_instance["infill_lang_mask"])
        merged_dataset["outcomes"].append(merged_instance["outcomes"])
        merged_dataset["infill_outcomes_mask"].append(merged_instance["infill_outcomes_mask"])                
                    
    return get_dataset(merged_dataset)


if __name__ == '__main__':
    
    long_lang_features, long_outcomes = read_dlatk_data()
    
    if 'embeddings_names' in long_lang_features:
        embeddings_names = long_lang_features.pop('embeddings_names')
        long_lang_features = get_dataset(long_lang_features)
    
    if 'outcomes_names' in long_outcomes:
        outcomes_names = long_outcomes.pop('outcomes_names')
        long_outcomes = get_dataset(long_outcomes)

    print ("Long Lang features and Long Outcomes read.")
    # NOTE: This data is not already split for train/ val/ test. Perform filtering and train-val split here
    
    # CHECK for user 3586 before and after creating mask
    long_lang_features = long_lang_features.map(create_lang_mask)
    print ("Lang mask pattern created.")
    
    long_outcomes = long_outcomes.map(create_outcomes_mask)
    print ("Outcomes mask pattern created.")    
    
    merged_dataset = merge_datasets(long_lang_features, long_outcomes)
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
    
    merged_dataset = merged_dataset.map(lambda x: truncate_sequences(x, 89))
    
    # Filter to sequences with at least 40 time_ids with either language or outcomes
    # To do that, add infill_lang_mask and infill_outcomes_mask to the dataset and filter if the number of 1s/2s in the mask is less than 40. 
    # Only consider the first outcome_index of infill_outcomes_mask
    def filter_sequences(instance):
        lang_mask = [1 if i==0 else 0 for i in instance['infill_lang_mask']]
        outcomes_mask = [mask[0] for mask in instance['infill_outcomes_mask']]
        summed_mask = [(l+o)>0 for l, o in zip(lang_mask, outcomes_mask)]
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
        outcomes = [outcome[0] for outcome, mask in zip(instance['outcomes'], instance['infill_outcomes_mask']) if mask[0]==1]
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
    
    # merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/todayPCL_selfreport_PCL_1_days_ahead_max90days_v3_40combined_5fold')
    # merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/todayPCL_selfreport_PCL_1_days_ahead_max90days_v4_40combined_5fold')
    merged_dataset.save_to_disk('/cronus_data/avirinchipur/ptsd_stop/forecasting/datasets/PCLsubscales_selfreport_PCL_1_days_ahead_max90days_v4_40combined_5fold')