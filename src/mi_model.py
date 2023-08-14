#################################
# The recurrent class has the recurrent model definition and forward() method. This is used to predict the seq class with each message.
# Early prediction is used to train the recurrent model with multi instance training.
#################################
#################################
####### IMPORTS #################
from pprint import pprint

# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (
#     f1_score, recall_score, precision_score, roc_auc_score,
#     mean_squared_error, mean_absolute_error, r2_score
# )
# from scipy.stats import pearsonr

import numpy as np
import torch
import torch.nn as nn
#################################

#################################
####### VARIABLES ###############
torch.manual_seed(42)
np.random.seed(42)
#################################

class recurrent(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0, bidirectional=False):
        super(recurrent, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        
        self.model = []
        
        self.model.append(nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, \
                                 dropout = self.dropout, bidirectional = self.bidirectional, batch_first=True))
        
        multiplier = (bidirectional + 1)
        self.model.append(nn.Linear(self.hidden_size*multiplier, self.num_classes))
        
        self.model = nn.ModuleList(self.model)
        
        
    def forward(self, input_rep, hidden_rep=None, mask=None, predict_last_valid_hidden_state=True):
        """
            input_rep: (batch_size, seq_len, input_size)
            hidden_rep: (num_layers * num_directions, batch_size, hidden_size)
            mask: (batch_size, seq_len) of type bool. 1 if the token is valid, 0 if not.
            predict_last_valid_hidden_state: If True, then the last valid timestep's hidden state from each instance is used for prediction. Else, all timesteps' hidden states are predicted.
        """
        # TODO: Check if the masking works correctly to perform MI prediction
        if mask is not None:
            # assert isinstance(mask, torch.BoolTensor), "Mask should be of type BoolTensor"
            assert mask.shape == torch.Size(input_rep.shape[:2]), "Mask shape should be (batch_size, seq_len)"
        
        output_rep = input_rep
        for layer in self.model:
            
            if mask is not None:
                # Alter method: use op.masked_fill()
                output_rep = output_rep * mask.unsqueeze(-1)
                
            if isinstance(layer, nn.GRU):
                output_rep, hidden_rep = layer(output_rep) if hidden_rep is None else layer(output_rep, hidden_rep)
            elif isinstance(layer, nn.Linear):
                if predict_last_valid_hidden_state: # Only predict for last valid timestep's hidden state
                    pos_mask = torch.zeros(mask.shape, device = mask.device)
                    if mask is not None: # If mask is not None, then we need to take the last VALID hidden state of each sequence
                        idx = torch.sum(mask, dim=1) - 1
                    else: # If mask is None, then we need to take the last hidden state of each sequence
                        idx = torch.tensor([-1]*output_rep.shape[0], device=output_rep.device)
                    pos_mask[torch.arange(output_rep.shape[0]), idx] = 1
                    output_rep = (output_rep*pos_mask.unsqueeze(-1)).sum(dim=1)
                else: # Predict for all timesteps' hidden states; Zero out the hidden states using mask if available
                    if mask is not None:
                        output_rep = (output_rep*mask.unsqueeze(-1))
                output = layer(output_rep)
        
        return output
#################################
