import argparse
from dataclasses import dataclass, field
import random
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
            seq_num_field: str=field(metadata={'help': 'sequence number field name'}, default='seq_num')
            feature_table: str=field(metadata={'help': 'feature table name'})
            outcome_table: str=field(metadata={'help': 'outcome table name'})
            outcome_field: str=field(metadata={'help': 'outcome field name'})
        
        Usage
        -----
            dlatk_data_getter = DLATKDataGetter(msg_table='emi_2016_2017', feature_table='emi_2016_2017_features', outcome_table='emi_2016_2017_outcomes', outcome_field='outcome')
            dataset_dict = dlatk_data_getter.combine_features_and_outcomes()
            dataset_dict = dlatk_data_getter.train_test_split(dataset_dict, test_ratio=0.15)
        
    """
    msg_table: str=field(metadata={'help': 'table name'})
    feature_table: str=field(metadata={'help': 'feature table name'})
    outcome_table: str=field(metadata={'help': 'outcome table name'})
    outcome_field: str=field(metadata={'help': 'outcome field name'})

    db: str=field(metadata={'help': 'database name'}, default="EMI")
    messageid_field: str=field(metadata={'help': 'message id field name'}, default='message_id')
    message_field: str=field(metadata={'help': 'message field name'}, default='message')
    correl_field: str=field(metadata={'help': 'correlation field name'}, default='seq_id')
    group_freq_thresh: int=field(metadata={'help': 'group frequency threshold'}, default=0)
    seq_num_field: str=field(metadata={'help': 'sequence number field name'}, default='seq_num')
    
    
    def __post_init__(self) -> None:
        
        assert self.msg_table is not None, "msg_table must be specified"
        assert self.feature_table is not None, "feature_table must be specified"
        assert self.outcome_table is not None, "outcome_table must be specified"
        assert self.outcome_field is not None, "outcome_field must be specified"
        
        self.args = argparse.Namespace(**self.__dict__)
    
    
    def get_features(self) -> dict:
        """
            Get query_id and embeddings from the feature table
        """
        fg = FeatureGetter(corpdb=self.args.db, corptable=self.args.msg_table, correl_field=self.args.messageid_field, featureTable=self.args.feature_table)
        gns = fg.getGroupNormsWithZeros()
        gns_dict = dict()
        for msg_id in gns[0].keys():
            gns_dict[msg_id] = []
            for feat in gns[1]:
                gns_dict[msg_id].append(gns[0][msg_id][feat])
                
        return gns_dict
    
    
    def get_msgid_seqid_seqnum_mapping(self):
        """
            Get a mapping of sequence ids, message ids and its sequence number 
        """
        fg = FeatureGetter(corpdb=self.args.db, corptable=self.args.msg_table, correl_field=self.args.messageid_field, featureTable=self.args.feature_table)
        #TODO: Add group_freq_thresh
        #TODO: Change the query below to be more generic
        sql = fg.qb.create_select_query(self.args.msg_table).set_fields(['message_id', 'seq_id', 'day_number'])
        print (sql.toString())
        
        msgid_seqid_mapping, msgid_seqnum_mapping = dict(), dict()
        for msg_id, seq_id, seq_num in sql.execute_query():
            if msg_id not in msgid_seqid_mapping: msgid_seqid_mapping[msg_id] = seq_id
            if msg_id not in msgid_seqnum_mapping: msgid_seqnum_mapping[msg_id] = seq_num
        
        print (f"Number of messages: {len(msgid_seqid_mapping)}")
        return msgid_seqid_mapping, msgid_seqnum_mapping


    def get_outcomes(self) -> dict:
        """
            Get the outcome values for the sequence ids
        """
        og = OutcomeGetter(corpdb=self.args.db, corptable=self.args.msg_table, correl_field=self.args.correl_field, outcome_table=self.args.outcome_table, outcome_value_fields=self.args.outcome_field)
        outcomes = og.getGroupAndOutcomeValues()
        outcomes_dict = dict()
        for seq_id, outcome_val in outcomes:
            outcomes_dict[seq_id] = outcome_val
        
        return outcomes_dict 

        
    def combine_features_and_outcomes(self) -> dict:
        """
            Combine the features and outcomes into a single dictionary.
            Returns a dictionary of the following format:
            {
                id: [seq_id1, seq_id2, ...],
                seq_num: [[seq_num1, seq_num2, ...], [seq_num1, seq_num2, ...], ...],
                embeddings: [[[emb1, emb2, ...], [emb1, emb2, ...], ...], [[emb1, emb2, ...], [emb1, emb2, ...], ...], ...],
                labels: [label1, label2, ...],
                message_ids: [[msg_id1, msg_id2, ...], [msg_id1, msg_id2, ...], ...]
            }
            
        """
        gns_dict = self.get_features()
        outcomes_dict = self.get_outcomes()
        msgid_seqid_mapping, msgid_seqnum_mapping = self.get_msgid_seqid_seqnum_mapping()
        
        seqid_msgid_mapping = dict()
        for msg_id, seq_id in msgid_seqid_mapping.items():
            if seq_id not in seqid_msgid_mapping: seqid_msgid_mapping[seq_id] = []
            temp = (msgid_seqnum_mapping[msg_id], msg_id, gns_dict[msg_id])
            seqid_msgid_mapping[seq_id].append(temp)
        
        for seq_id, msgid_seqnum_list in seqid_msgid_mapping.items():
            seqid_msgid_mapping[seq_id] = sorted(msgid_seqnum_list, key=lambda x: x[0])
        
        seqids = set(msgid_seqid_mapping.values())
        dataset_dict = dict(id=[], seq_num=[], embeddings=[], labels=[], message_ids=[])
        
        for seq_id in seqids:
            dataset_dict['id'].append(seq_id)
            dataset_dict['seq_num'].append([x[0] for x in seqid_msgid_mapping[seq_id]])
            dataset_dict['message_ids'].append([x[1] for x in seqid_msgid_mapping[seq_id]])
            dataset_dict['embeddings'].append([[x[2] for x in seqid_msgid_mapping[seq_id]]])
            # For multi instance learning, the outcome would be a list of labels for each instance (i.e., seq_num) of the sequence
            dataset_dict['labels'].append([outcomes_dict[seq_id]]*len(seqid_msgid_mapping[seq_id]))
            # dataset_dict['labels'].append(outcomes_dict[seq_id])
            
        return dataset_dict

    
    def train_test_split(self, dataset_dict:dict, test_ratio:float=0.2) -> dict:
        """
            Split the dataset into train and test based on the test_ratio
        """
        assert test_ratio > 0 and test_ratio < 1, "test_ratio must be between 0 and 1"
        
        all_idx = list(range(len(dataset_dict['embeddings'])))
        train_idx = random.sample(all_idx, int((1-test_ratio)*len(all_idx)))
        test_idx = list(set(all_idx) - set(train_idx))
        
        train_datadict, test_datadict = dict(), dict()
        for key in dataset_dict:
            train_datadict[key] = [dataset_dict[key][idx] for idx in train_idx]
            test_datadict[key] = [dataset_dict[key][idx] for idx in test_idx]
                
        return dict(train=train_datadict, test=test_datadict)