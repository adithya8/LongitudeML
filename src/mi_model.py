#################################
# The recurrent class has the recurrent model definition and forward() method. This is used to predict the seq class with each message.
# Early prediction is used to train the recurrent model with multi instance training.
#################################
#################################
####### IMPORTS #################
import torch
import torch.nn as nn
#################################

#################################
####### VARIABLES ###############
# TODO: Move this to the main script
torch.manual_seed(42)
#################################

class recurrent(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0, bidirectional=False, output_dropout=0.0):
        super(recurrent, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        
        # TODO: Positional encoding
        self.model = []
        self.output_dropout = nn.Dropout(output_dropout)
        self.ln = nn.LayerNorm(input_size)
        
        #TODO: Decouple the GRU layer from the linear layer. 
        self.model.append(nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, \
                                 dropout = self.dropout, bidirectional = self.bidirectional, batch_first=True))
        
        multiplier = (bidirectional + 1)
        if self.num_classes<=2: # For binary classification or regression, we need only one output node
            self.model.append(nn.Linear(self.hidden_size*multiplier, 1))
        else:
            self.model.append(nn.Linear(self.hidden_size*multiplier, self.num_classes))
        
        self.model = nn.ModuleList(self.model)
        
        
    def forward(self, embeddings:torch.Tensor, hidden_rep:torch.Tensor=None, mask:torch.Tensor=None, predict_last_valid_hidden_state:bool=True):
        """
            embeddings: (batch_size, seq_len, input_size)
            hidden_rep: (num_layers * num_directions, batch_size, hidden_size)
            mask: (batch_size, seq_len) of type bool. 1 if the token is valid, 0 if not.
            predict_last_valid_hidden_state: If True, then the last valid timestep's hidden state from each instance is used for prediction. Else, all timesteps' hidden states are predicted.
        """
        if mask is not None:
            # assert isinstance(mask, torch.BoolTensor), "Mask should be of type BoolTensor" #TODO: This line throws assertion error although the mask is of type BoolTensor
            assert mask.shape == torch.Size(embeddings.shape[:2]), "Mask shape should be (batch_size, seq_len). Got {} for mask and {} for embeddings".format(mask.shape, embeddings.shape)
        
        output_rep = self.ln(embeddings)
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
                    if mask is not None: output_rep = (output_rep*mask.unsqueeze(-1))  
                output = layer(self.output_dropout(output_rep)).squeeze(-1)
                
        
        return output
    
    
class AutoRegressiveTransformer(nn.Module):
    """
        
    """
    def __init__(self, input_size, hidden_size, num_classes, num_layers=1, dropout=0.0, bidirectional=False, output_dropout=0.0, num_heads=1):
        super(AutoRegressiveTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_heads = num_heads
        self.init_model()
    
    def init_model(self):
        self.model = []
        self.output_dropout_layer = nn.Dropout(self.output_dropout)
        self.ln = nn.LayerNorm(self.input_size)
        
        for _ in range(self.num_layers):
            self.model.append(nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_heads, 
                                       dim_feedforward=self.hidden_size, dropout=self.dropout, batch_first=True))
        
        if self.num_classes<=2: # For binary classification or regression, we need only one output node
            self.model.append(nn.Linear(self.hidden_size, 1))
        else:
            self.model.append(nn.Linear(self.hidden_size, self.num_classes))
        
        self.model = nn.ModuleList(self.model)
    
    def forward(self, embeddings:torch.Tensor, mask:torch.Tensor=None, predict_last_valid_hidden_state:bool=True):
        """
            embeddings: (batch_size, seq_len, input_size)
            mask: (batch_size, seq_len) of type bool. 1 if the token is valid, 0 if not.
            predict_last_valid_hidden_state: If True, then the last valid timestep's hidden state from each instance is used for prediction. Else, all timesteps' hidden states are predicted.
        """
        
        if mask is not None:
            # assert isinstance(mask, torch.BoolTensor), "Mask should be of type BoolTensor" #TODO: This line throws assertion error although the mask is of type BoolTensor
            assert mask.shape == torch.Size(embeddings.shape[:2]), "Mask shape should be (batch_size, seq_len). Got {} for mask and {} for embeddings".format(mask.shape, embeddings.shape)
            
        output_rep = self.ln(embeddings)
        for layer in self.model:
            
            if mask is not None:
                # Alter method: use op.masked_fill()
                output_rep = output_rep * mask.unsqueeze(-1)
                
            if isinstance(layer, nn.TransformerEncoderLayer):
                output_rep = layer(output_rep)
            elif isinstance(layer, nn.Linear):
                if predict_last_valid_hidden_state:
                    pos_mask = torch.zeros(mask.shape, device = mask.device)
                    if mask is not None:
                        idx = torch.sum(mask, dim=1) - 1
                    else:
                        idx = torch.tensor([-1]*output_rep.shape[0], device=output_rep.device)
                    pos_mask[torch.arange(output_rep.shape[0]), idx] = 1
                    output_rep = (output_rep*pos_mask.unsqueeze(-1)).sum(dim=1)
                else:
                    if mask is not None: output_rep = (output_rep*mask.unsqueeze(-1))
                output = layer(self.output_dropout(output_rep)).squeeze(-1)
                
        return output
        

        
#################################
