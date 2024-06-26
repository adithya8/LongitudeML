import argparse
from dataclasses import dataclass, field
from functools import reduce
import random
from typing import List, Union
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from dlatk.featureGetter import FeatureGetter
from dlatk.outcomeGetter import OutcomeGetter

@dataclass
class DLATKDataGetter:
    """
        Class to process the DLATK features table and outcome table into data dictionary
        Input
        -----
            db: str=field(metadata={'help': 'database name'}, default="EMI")
            msg_table: str=field(metadata={'help': 'table name'})
            messageid_field: str=field(metadata={'help': 'message id field name'}, default='message_id')
            message_field: str=field(metadata={'help': 'message field name'}, default='message')
            correl_field: str=field(metadata={'help': 'correlation field name'}, default='seq_id')
            group_freq_thresh: int=field(metadata={'help': 'group frequency threshold'}, default=0)
            timeid_field: str=field(metadata={'help': 'time index field name'}, default='time_id')
            feature_tables: str=field(metadata={'help': 'feature table name'})
            outcome_table: str=field(metadata={'help': 'outcome table name'})
            outcome_field: str=field(metadata={'help': 'outcome field name'})
        
        Usage
        -----
            >>> dlatk_data_getter = DLATKDataGetter(msg_table='emi_2016_2017', feature_tables='emi_2016_2017_features', outcome_table='emi_2016_2017_outcomes', outcome_field='outcome')
            >>> dataset_dict = dlatk_data_getter.combine_features_and_outcomes()
            >>> dataset_dict = dlatk_data_getter.train_test_split(dataset_dict, test_ratio=0.15)
        
    """
    msg_table: str=field(metadata={'help': 'table name'})
    feature_tables: Union[str, List[str]]=field(metadata={'help': 'feature tables name'})
    outcome_table: str=field(metadata={'help': 'outcome table name'})
    outcome_field: str=field(metadata={'help': 'outcome field name'})

    db: str=field(metadata={'help': 'database name'}, default="EMI")
    messageid_field: str=field(metadata={'help': 'message id field name'}, default='message_id') #TODO: add alias="queryid_field"
    message_field: str=field(metadata={'help': 'message field name'}, default='message') #TODO: add alias="query_field"
    correl_field: str=field(metadata={'help': 'correlation field name'}, default='seq_id')
    group_freq_thresh: int=field(metadata={'help': 'group frequency threshold'}, default=0)
    timeid_field: str=field(metadata={'help': 'sequence number field name'}, default='time_id')
    
    
    def __post_init__(self) -> None:
        
        assert self.msg_table is not None, "msg_table must be specified"
        assert self.feature_tables is not None, "feature_tables must be specified"
        assert self.outcome_table is not None, "outcome_table must be specified"
        assert self.outcome_field is not None, "outcome_field must be specified"
        
        self.args = argparse.Namespace(**self.__dict__)


    def get_features(self, where='') -> dict:
        """
            Get query_id and embeddings from the feature table
            Returns
            -------
                A dictionary of the following format:
                {
                    query_id1: [emb1, emb2, ...],
                    query_id2: [emb1, emb2, ...],
                    ...
                }
        """
        if isinstance(self.feature_tables, str): self.feature_tables = [self.feature_tables]
        fg = FeatureGetter(corpdb=self.args.db, corptable=self.args.msg_table, correl_field=self.args.messageid_field)
        temp_gns_dict = dict()
        for table in self.feature_tables:
            fg.featureTable = table
            gns, feats = fg.getGroupNormsWithZeros(where=where)
            table_gns_dict = dict()
            for query_id in gns.keys():
                table_gns_dict[query_id] = []
                for feat in feats:
                    table_gns_dict[query_id].append(gns[query_id][feat])
            temp_gns_dict[table] = table_gns_dict
                    
        common_qryids = []
        for table_gns_dict in temp_gns_dict.values(): 
            common_qryids.append(set(table_gns_dict.keys()))
        common_qryids = reduce(lambda x, y: x.intersection(y), common_qryids)
        
        gns_dict = dict()
        # concat the embeddings from all the feature tables only for the common query ids
        for query_id in common_qryids:
            gns_dict[query_id] = []
            for table in temp_gns_dict.keys():
                gns_dict[query_id].extend(temp_gns_dict[table][query_id])

        return gns_dict


    def get_qryid_seqid_timeids_mapping(self):
        """
            Get a mapping of sequence ids, query ids and its time idx. 
            The seq_id is unique to a sequnce of queries. The query ids map to a time_id which represents ordering of the queries in the sequence.
            Returns
            -------
                qryid_seqid_mapping: A dictionary of the following format:
                {
                    query_idx1: seq_idx1,
                    query_idx2: seq_idx2,
                    ...
                }
                qryid_timeids_mapping: A dictionary of the following format:
                {
                    query_idx1: time_idx1,
                    query_idx2: time_idx2,
                    ...
                }
                longtype_encoder: A dictionary of the following format:
                {
                    qryid_mappings: {
                        query_idx1: long_query_idx1,
                        query_idx2: long_query_idx2,
                        ...
                    },
                    seqid_mappings: {
                        seq_idx1: long_seq_idx1,
                        seq_idx2: long_seq_idx2,
                        ...
                    },
                    qryid_mappings_rev: {
                        long_query_idx1: query_idx1,
                        long_query_idx2: query_idx2,
                        ...
                    },
                    seqid_mappings_rev: {
                        long_seq_idx1: seq_idx1,
                        long_seq_idx2: seq_idx2,
                        ...
                    }
                } 
        """
        fg = FeatureGetter(corpdb=self.args.db, corptable=self.args.msg_table, correl_field=self.args.messageid_field)
        #TODO: Add group_freq_thresh/ min query per sequence
        #TODO: Add where to filter to only those sequences that have features
        #TODO: Change the query below to be more generic. This works only for DS4UD anilsson data
        sql = fg.qb.create_select_query(self.args.msg_table).set_fields([self.args.messageid_field, self.correl_field, self.timeid_field]).where("{} IS NOT NULL".format(self.timeid_field))
        print (sql.toString())
        
        qryid_seqid_mapping, qryid_timeids_mapping = dict(), dict()
        for qry_id, seq_id, time_idx in sql.execute_query():
            if qry_id not in qryid_seqid_mapping: qryid_seqid_mapping[qry_id] = seq_id
            if qry_id not in qryid_timeids_mapping: qryid_timeids_mapping[qry_id] = time_idx
        
        # Store the original query_id/seq_id mapping with corresponding long datatype query_id/seq_id. This step is necessary since strings can't be parsed into torch tensors directly.
        longtype_encoder = dict(qryid_mappings=dict(), seqid_mappings=dict(), qryid_mappings_rev=dict(), seqid_mappings_rev=dict())
        for qry_id, seq_id in qryid_seqid_mapping.items():
            if isinstance(qry_id, str):
                longtype_encoder['qryid_mappings'][qry_id] = qry_id.isdigit() if qry_id.isdigit() else len(longtype_encoder['qryid_mappings']) + 1
            else:
                longtype_encoder['qryid_mappings'][qry_id] = qry_id
            if isinstance(seq_id, str):
                if seq_id not in longtype_encoder['seqid_mappings']: 
                    longtype_encoder['seqid_mappings'][seq_id] = seq_id.isdigit() if seq_id.isdigit() else len(longtype_encoder['seqid_mappings']) + 1
            else:
                longtype_encoder['seqid_mappings']: longtype_encoder['seqid_mappings'][seq_id] = seq_id
            longtype_encoder['qryid_mappings_rev'][longtype_encoder['qryid_mappings'][qry_id]] = qry_id
            longtype_encoder['seqid_mappings_rev'][longtype_encoder['seqid_mappings'][seq_id]] = seq_id                
        print (f"Number of messages: {len(qryid_seqid_mapping)}")
        
        return qryid_seqid_mapping, qryid_timeids_mapping, longtype_encoder


    def get_outcomes(self, outcome_field:str=None, correl_field:str=None, where:str='') -> dict:
        """
            Get the outcome values for the sequence ids.
            Returns
            -------
                A dictionary of the following format:
                {
                    seq_id1: outcome_val1,
                    seq_id2: outcome_val2,
                    ...
                }
        """
        assert correl_field in [self.messageid_field, self.correl_field, None], "correl_fielf for outcome should be either messageid_field or correl_field"
        
        outcome_field = self.args.outcome_field if outcome_field is None else outcome_field
        if correl_field is None: correl_field = self.args.correl_field
        og = OutcomeGetter(corpdb=self.args.db, corptable=self.args.msg_table, correl_field=correl_field, outcome_table=self.args.outcome_table)
        outcomes = og.getGroupAndOutcomeValues(outcomeField=outcome_field, where=where)
        outcomes_dict = dict()
        # row_id since it can be either message_id or seq_id
        for row_id, outcome_val in outcomes:
            outcomes_dict[row_id] = outcome_val
        
        return outcomes_dict 

        
    def combine_features_and_outcomes(self, outcomes_correl_field:str=None, features_where:Union[str, List[str]]='', outcomes_where:str='') -> dict:
        """
            Combine the features and outcomes into a single dictionary.
            Returns a dictionary of the following format:
            {
                seq_idx: [seq_id1, seq_id2, ...],
                time_ids: [[time_idx1, time_idx2, ...], [time_idx1, time_idx2, ...], ...],
                embeddings: [[[emb1, emb2, ...], [emb1, emb2, ...], ...], [[emb1, emb2, ...], [emb1, emb2, ...], ...], ...],
                labels: [[label1, label1, ...], [label2, label2, ...], ...],
                query_ids: [[qry_idx1, qry_idx2, ...], [qry_idx1, qry_idx2, ...], ...]
            }
        """
        gns_dict = self.get_features(where=features_where)
        outcomes_dict = self.get_outcomes(where=outcomes_where, correl_field=outcomes_correl_field)
        qryid_seqid_mapping, qryid_timeids_mapping, longtype_encoder = self.get_qryid_seqid_timeids_mapping()
        
        # The following snippet:
        # DESC: maps seq_id to a list of (time_idx, qry_id, emb). Example - seq_id1: [(time_idx1_1, qry_id1, emb_list1), (time_idx1_2, qry_id2, emb_list2) ...]
        seqid_qryid_mapping = dict() 
        for qry_id, seq_id in qryid_seqid_mapping.items():
            qry_id_long, seq_id_long = longtype_encoder['qryid_mappings'][qry_id], longtype_encoder['seqid_mappings'][seq_id]
            if seq_id_long not in seqid_qryid_mapping: seqid_qryid_mapping[seq_id_long] = []
            temp = (qryid_timeids_mapping[qry_id], qry_id_long, gns_dict[qry_id])
            seqid_qryid_mapping[seq_id_long].append(temp)
        
        # DESC: Sort the qryids by time_idx for each seq_id
        for seq_id_long, qryid_timeids_list in seqid_qryid_mapping.items():
            seqid_qryid_mapping[seq_id_long] = sorted(qryid_timeids_list, key=lambda x: x[0])
        
        seqids = set(qryid_seqid_mapping.values())
        
        if outcomes_correl_field == self.messageid_field:
            # reformat outcomes_dict into nested dict with seq_id as key and {queryid: outcome} as value
            outcomes_dict_reformatted = dict()
            for qry_id in outcomes_dict.keys():
                seq_id = qryid_seqid_mapping[qry_id]
                if seq_id not in outcomes_dict_reformatted: outcomes_dict_reformatted[seq_id] = dict()
                outcomes_dict_reformatted[seq_id][qry_id] = outcomes_dict[qry_id]
            outcomes_dict = outcomes_dict_reformatted
            
        dataset_dict = dict(seq_idx=[], time_ids=[], embeddings=[], labels=[], query_ids=[])
        
        # Populate the dataset_dict
        for seq_id in seqids:
            seq_id_long = longtype_encoder['seqid_mappings'][seq_id]
            if seq_id not in outcomes_dict:
                print (f"Seq_id {seq_id} not found in outcomes_dict. Skipping!!")
                continue
            temp_qry_ids = [longtype_encoder['qryid_mappings_rev'][x[1]] for x in seqid_qryid_mapping[seq_id_long]]
            
            if outcomes_correl_field == self.messageid_field:
                temp_embs = []
                temp_labels = []
                temp_time_ids = []
                temp_query_ids = []
                for idx, qry_id in enumerate(temp_qry_ids):
                    if qry_id not in outcomes_dict[seq_id]:
                        print (f"Query_id {qry_id} not found in outcomes_dict for seq_id {seq_id}. Skipping!!")
                        continue
                    temp_query_ids.append(qry_id)
                    temp_embs.append(seqid_qryid_mapping[seq_id_long][idx])
                    temp_labels.append(outcomes_dict[seq_id][qry_id])
                    temp_time_ids.append(seqid_qryid_mapping[seq_id_long][idx][0])

                if temp_query_ids is not None:
                    dataset_dict['seq_idx'].append(seq_id_long)
                    dataset_dict['time_ids'].append(temp_time_ids)
                    dataset_dict['query_ids'].append(temp_query_ids)
                    dataset_dict['embeddings'].append([temp_embs])
                    dataset_dict['labels'].append(temp_labels) 
            else:
                dataset_dict['seq_idx'].append(seq_id_long)
                dataset_dict['time_ids'].append([x[0] for x in seqid_qryid_mapping[seq_id_long]])
                dataset_dict['query_ids'].append([x[1] for x in seqid_qryid_mapping[seq_id_long]])
                dataset_dict['embeddings'].append([[x[2] for x in seqid_qryid_mapping[seq_id_long]]])
                # For multi instance learning, the outcome would be a list of labels for each instance (i.e., time_idx) of the sequence
                dataset_dict['labels'].append([outcomes_dict[seq_id]]*len(seqid_qryid_mapping[seq_id_long]))
                            
        return dataset_dict, longtype_encoder


    def clamp_sequence_length(self, dataset_dict:dict, min_seq_len:int=3, max_seq_len:int=14, retain:str="last") -> dict:
        """
            Clamp the sequence length to min_seq_len and max_seq_len (min and max bounds included)
            retain: str - retains either the first or the last max_seq_len instances of the sequence. Default: last 
        """
        assert retain in ["first", "last"], "retain must be either 'first' or 'last'"
        
        bool_mask = []
        for i in range(len(dataset_dict['embeddings'])):
            if len(dataset_dict['embeddings'][i][0])<min_seq_len:  
                bool_mask.append(0)
            else:
                if len(dataset_dict['embeddings'][i][0])>max_seq_len:
                    if retain == "first":
                        # TODO: Choose the first max_seq_len instances of the sequence, such that last instance time_id - first instance time_id <= max_seq_len and last instance time_id - first instance time_id >= min_seq_len
                        for key in dataset_dict.keys():
                            if key == "embeddings": 
                                dataset_dict[key][i] = [dataset_dict[key][i][0][:max_seq_len]]
                            elif isinstance(dataset_dict[key][i], List): 
                                dataset_dict[key][i] = dataset_dict[key][i][:max_seq_len]
                    else:
                        # TODO: Choose the last max_seq_len instances of the sequence, such that last instance time_id - first instance time_id <= max_seq_len and last instance time_id - first instance time_id >= min_seq_len
                        for key in dataset_dict.keys():
                            if key == "embeddings": 
                                dataset_dict[key][i] = [dataset_dict[key][i][0][-max_seq_len:]]
                            elif isinstance(dataset_dict[key][i], List): 
                                dataset_dict[key][i] = dataset_dict[key][i][-max_seq_len:]
                bool_mask.append(1)
        
        # apply mask
        for key in dataset_dict.keys():
            dataset_dict[key] = [dataset_dict[key][i] for i in range(len(dataset_dict[key])) if bool_mask[i]==1]
        
        return dataset_dict


    def train_test_split(self, dataset_dict:dict, test_ratio:float=0.2, val_ratio:float=0.0, stratify=None) -> dict:
        """
            Split the dataset into train and test based on the test_ratio
        """
        assert test_ratio > 0 and test_ratio < 1, "test_ratio must be between 0 and 1"
        assert val_ratio >= 0 and val_ratio < 1, "val_ratio must be between 0 and 1"
        assert test_ratio + val_ratio < 1, "test_ratio + val_ratio must be less than 1"
        
        train_ratio = 1 - test_ratio - val_ratio
        all_idx = list(range(len(dataset_dict['embeddings'])))
        labels=None
        if stratify:
            labels = list(map(lambda x: x[-1], dataset_dict['labels']))
            if isinstance(labels[0], float): labels = np.minimum((np.argsort(labels)/len(labels))//10, 4)  #Making labels discrete to stratify
        if test_ratio>0: train_idx, test_idx = train_test_split(all_idx, test_size=test_ratio, stratify=labels, random_state=52)
        if stratify: labels = [labels[idx] for idx in train_idx]
        if val_ratio>0: train_idx, val_idx = train_test_split(train_idx, test_size=(val_ratio/train_ratio), stratify=labels, random_state=52)
             
        train_datadict, val_datadict, test_datadict = dict(), dict(), dict()
        for key in dataset_dict:
            train_datadict[key] = [dataset_dict[key][idx] for idx in train_idx]
            test_datadict[key] = [dataset_dict[key][idx] for idx in test_idx]
            if val_ratio>0: val_datadict[key] = [dataset_dict[key][idx] for idx in val_idx]
            
        return dict(train_data=train_datadict, val_data=val_datadict, test_data=test_datadict)
    
    
    def train_test_split(self, dataset_dict:dict, longtype_encoder:dict, test_fold_column:str=None, val_fold_column:str=None) -> dict:
        """
            Splits the dataset into train and test based on the test_fold_column in the outcomes table.
        """
        
        if test_fold_column: seqid_testset_mappings = self.get_outcomes(outcome_field=test_fold_column)
        if val_fold_column: seqid_valset_mappings = self.get_outcomes(outcome_field=val_fold_column)
        long_seq_ids = dataset_dict['seq_idx']
        train_idx, val_idx, test_idx = [], [], []
        for idx, long_seqid in enumerate(long_seq_ids):
            seqid = longtype_encoder['seqid_mappings_rev'][long_seqid]
            if test_fold_column and seqid in seqid_testset_mappings:
                test_idx.append(idx)
            elif val_fold_column and seqid in seqid_valset_mappings:
                val_idx.append(idx)
            else:
                train_idx.append(idx)
        
        train_datadict, val_datadict, test_datadict = dict(), dict(), dict()
        for key in dataset_dict:
            train_datadict[key] = [dataset_dict[key][idx] for idx in train_idx]
            if test_fold_column: test_datadict[key] = [dataset_dict[key][idx] for idx in test_idx]
            if val_fold_column: val_datadict[key] = [dataset_dict[key][idx] for idx in val_idx]

        return dict(train_data=train_datadict, val_data=val_datadict, test_data=test_datadict)        


    # def n_fold_split(self, dataset_dict:dict, folds:int=5, stratify=None) -> dict:
    #     """
    #         Splits the dataset into n folds
    #     """
    #     assert folds > 1, "folds must be greater than 1"
        
    #     # if dataset is split into train and test, perform the nfold split on train set, else perform on the entire dataset
    #     folds_idx = []
    #     all_idx = list(range(len(dataset_dict['train_data']['embeddings']))) if 'train_data' in dataset_dict else list(range(len(dataset_dict['embeddings'])))
    #     labels=None
    #     if stratify:
    #         labels = list(map(lambda x: x[-1], dataset_dict['train_data']['labels'])) if 'train_data' in dataset_dict else list(map(lambda x: x[-1], dataset_dict['labels']))
    #         if isinstance(labels[0], float): labels = np.minimum((np.argsort(labels)/len(labels))//10, 4)
            
    #         skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    #         for fold, (_, test_idx) in enumerate(skf.split(all_idx, labels)):
    #             folds_idx.extend([(fold, idx) for idx in test_idx])
            
    #         folds_idx = sorted(folds_idx, key=lambda x: x[1])
    #         folds_idx = [x[0] for x in folds_idx]
    #     else:
    #         random.seed(42)
    #         folds_idx = random.sample(range(folds), k=len(all_idx))
        
    #     if 'train_data' in dataset_dict:
    #         dataset_dict['train_data']['folds'] = folds_idx
    #     else:
    #         dataset_dict['folds'] = folds_idx
        
    #     return dataset_dict
    
    
    def n_fold_split(self, dataset_dict:dict, longtype_encoder:dict, fold_column:str) -> dict:
        """
            Splits the dataset into n folds based on the fold column in the outcomes tables.
        """
        
        folds_idx = []
        seqid_fold_mappings = self.get_outcomes(outcome_field=fold_column)
        long_seq_ids = dataset_dict['train_data']['seq_idx'] if 'train_data' in dataset_dict else dataset_dict['seq_idx'] 
        for long_seqid in long_seq_ids:
            seqid = longtype_encoder['seqid_mappings_rev'][long_seqid]
            longseqid_fold = seqid_fold_mappings[seqid]
            folds_idx.append(longseqid_fold)
        
        if 'train_data' in dataset_dict:
            dataset_dict['train_data']['folds'] = folds_idx
        else:
            dataset_dict['folds'] = folds_idx
        
        return dataset_dict