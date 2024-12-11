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
    def __init__(self, input_size, hidden_size, num_classes, num_outcomes=1, num_layers=1, dropout=0.0, bidirectional=False, output_dropout=0.0):
        super(recurrent, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_outcomes = num_outcomes
        
        # TODO: Positional encoding
        self.model = []
        self.output_dropout = nn.Dropout(output_dropout)
        self.ln = nn.LayerNorm(input_size)
        
        #TODO: Decouple the GRU layer from the linear layer. 
        self.model.append(nn.GRU(input_size = self.input_size, hidden_size = self.hidden_size, num_layers = self.num_layers, \
                                 dropout = self.dropout, bidirectional = self.bidirectional, batch_first=True))
        
        multiplier = (bidirectional + 1)
        if self.num_classes<=2: # For binary classification or regression, we need only one output node
            self.model.append(nn.Linear(self.hidden_size*multiplier, 1*self.num_outcomes))
        else:
            self.model.append(nn.Linear(self.hidden_size*multiplier, self.num_classes*self.num_outcomes))
        
        self.model = nn.ModuleList(self.model)
        
        
    def forward(self, embeddings:torch.Tensor, hidden_rep:torch.Tensor=None, mask:torch.Tensor=None, predict_last_valid_hidden_state:bool=True, **kwargs):
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
    

class PositionalEncoding(nn.Module):
    """
        Adds learnable positional encoding to the input embeddings.
    """
    def __init__(self, d_model:int, max_len:int):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.init_pe()
    
    def init_pe(self):
        self.positional_embedding = torch.nn.Embedding(self.max_len, self.d_model)
        
    def forward(self, x):
        x = x + self.positional_embedding(torch.arange(x.shape[1], device=x.device))
        return x

    
class AutoRegressiveTransformer(nn.Module):
    """
        
    """
    def __init__(self, input_size:int, hidden_size:int, num_classes:int, num_outcomes:int=1, num_layers:int=1, 
                 dropout:float=0.0, bidirectional:bool=False, output_dropout:float=0.0, num_heads:int=4, max_len:int=120):
        super(AutoRegressiveTransformer, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.output_dropout = output_dropout
        self.dropout = dropout
        self.num_classes = num_classes
        self.num_outcomes = num_outcomes
        self.num_heads = num_heads
        self.max_len = max_len
        self.init_model()
    
    def init_model(self):
        self.model = []
        # init infill_embeddings of side (input_size, ) which can be learnt when embeddings for a timestep is missing
        # Perform Xavier initialization for the infill_embeddings.
        # std = gain * sqrt(2.0 / (fan_in + fan_out)); gain = 1.0, fan_in = 1, fan_out = self.input_size
        std = 1.0 * (2.0 / (1 + self.input_size))**0.5
        self.infill_embeddings = nn.Parameter(torch.normal(0, std**2, (self.input_size,)))        
        self.positional_encoding = PositionalEncoding(self.input_size, self.max_len)
        self.output_dropout_layer = nn.Dropout(self.output_dropout)
        # self.ln = nn.LayerNorm(self.input_size) # layernorm is already present in the TransformerEncoderLayer
        
        for _ in range(self.num_layers):
            # Use TransformerEncoderLayer for building Decoder only Transformer model by using is_causal=True
            # https://discuss.pytorch.org/t/nn-transformerdecoderlayer-without-encoder-input/183990/2
            self.model.append(nn.TransformerEncoderLayer(d_model=self.input_size, nhead=self.num_heads, 
                                       dim_feedforward=self.hidden_size, dropout=self.dropout, batch_first=True))
        
        if self.num_classes<=2: # For binary classification or regression, we need only one output node
            self.model.append(nn.Linear(self.input_size, 1*self.num_outcomes))
        else:
            self.model.append(nn.Linear(self.input_size, self.num_classes*self.num_outcomes))
        
        self.model = nn.ModuleList(self.model)
    
    def forward(self, embeddings:torch.Tensor, mask:torch.Tensor=None, predict_last_valid_hidden_state:bool=True, **kwargs):
        """
            embeddings: (batch_size, seq_len, input_size)
            mask: (batch_size, seq_len) of type bool. if mask = 1, then the query is was padded and should not be used for attention. If mask = 0, then the query is valid and should be used for attention.
            predict_last_valid_hidden_state: If True, then the last valid timestep's hidden state from each instance is used for prediction. Else, all timesteps' hidden states are predicted.
        """
        
        if mask is not None:
            # assert isinstance(mask, torch.BoolTensor), "Mask should be of type BoolTensor" #TODO: This line throws assertion error although the mask is of type BoolTensor
            assert mask.shape == torch.Size(embeddings.shape[:2]), "Mask shape should be (batch_size, seq_len). Got {} for mask and {} for embeddings".format(mask.shape, embeddings.shape)

        # Create infill mask first. Set to true for timesteps that were infilled whicch are timesteps with time_id != -1 and mask == 1. Else set to False. Create it in the same device as the embeddings
        infill_mask = torch.where((kwargs['time_ids']!=-1) & mask, True, False).to(embeddings.device)
        # adding infill_embeddings whereever the mask is True

        output_rep = embeddings + (infill_mask.unsqueeze(-1)*self.infill_embeddings)
        # adding positional encoding to the embeddings
        output_rep = self.positional_encoding(embeddings)
        # src_key_padding_mask only to mask out the padded tokens, i.e., timestep with time_ids = -1. 
        # Boolean src_key_padding_mask uses True to mask out the padded tokens. Floating uses -inf to mask out the padded tokens and 0.0 to keep the valid tokens.
        src_key_padding_mask_bool = torch.where(kwargs['time_ids']==-1, True, False)        
        # Generate a square mask for the src_mask parameter in the TransformerEncoderLayer. This should be of size (seq_len, seq_len)
        # This mask is used to mask out the future tokens in the sequence. For Boolean mask, True is used to mask out the future tokens, False is used to keep the present and past tokens. 
        # For float -inf is used to mask out the future tokens, 0s are used to keep the present and past tokens.
        src_mask_bool = ~torch.tril(torch.ones(embeddings.shape[1], embeddings.shape[1], device=embeddings.device)).bool()
        # src_mask_float = nn.Transformer.generate_square_subsequent_mask(sz=embeddings.shape[1], device=embeddings.device)
        for layer in self.model:                
            if isinstance(layer, nn.TransformerEncoderLayer):
                output_rep = layer(src=output_rep, is_causal=True, src_mask=src_mask_bool, src_key_padding_mask=src_key_padding_mask_bool)
            elif isinstance(layer, nn.Linear):
                if predict_last_valid_hidden_state:
                    pos_mask = torch.zeros(mask.shape, device = mask.device)
                    if mask is not None:
                        # Find the first False in mask from the right side for each instance in the batch
                        idx = torch.argmax(~torch.flip(mask, [1]), dim=1)
                        idx = mask.shape[1] - idx - 1
                        # idx = torch.sum(mask, dim=1) - 1
                    else:
                        idx = torch.tensor([-1]*output_rep.shape[0], device=output_rep.device)
                    pos_mask[torch.arange(output_rep.shape[0]), idx] = 1
                    output_rep = (output_rep*pos_mask.unsqueeze(-1)).sum(dim=1)
                output = layer(self.output_dropout_layer(output_rep))
        
        # if torch.isnan(output).any(): import pdb; pdb.set_trace()
        return output
#################################
