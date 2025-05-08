#################################
# The recurrent class has the recurrent model definition and forward() method. This is used to predict the seq class with each message.
# Early prediction is used to train the recurrent model with multi instance training.
#################################
#################################
####### IMPORTS #################
import torch
import torch.nn as nn
import math
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
        # Check if x.shape[1] < max_len, 
        # If yes, directly add the positional embeddings of arange(x.shape[1])
        # If no, find the remainder of (x.shape[1] - max_len), and add poistional embedding for 0-max_len timesteps of x to regular positional embeddings, for the remaining timesteps, find the timestep that produces the same remainder when dividing max_len by 7 and can also hold the remaining timesteps.
        if x.shape[1] <= self.max_len:
            x = x + self.positional_embedding(torch.arange(x.shape[1], device=x.device))
        else:
            remainder = x.shape[1] - self.max_len  
            start_position = self.max_len - remainder
            # TODO: This needs to be fixed. Not the entire batch will have the same max_len.
            while start_position%7 != self.max_len%7: start_position -= 1
            pos_embs_maxlen = self.positional_embedding(torch.arange(self.max_len, device=x.device))
            pos_embs_remainder = self.positional_embedding(torch.arange(start_position, start_position+remainder, device=x.device))
            x  = x + torch.cat([pos_embs_maxlen, pos_embs_remainder], dim=0)
        return x


class SinusoidalPositionalEncoding(nn.Module):
    """
        Adds sinusoidal positional encoding to the input embeddings.
    """
    def __init__(self, d_model:int, max_len:int):
        super(SinusoidalPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.init_pe()
    
    def init_pe(self):
        pe = torch.zeros(self.max_len, self.d_model)
        position = torch.arange(0, self.max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2)/self.d_model*(-math.log(10000.0)))
        pe[:, 0::2] = torch.sin(position*div_term)
        pe[:, 1::2] = torch.cos(position*div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        if x.shape[1] <= self.max_len:
            x = x + self.pe[:x.shape[1]]
        else:
            # add the positional embeddings for the first max_len timesteps 
            # and then add the positional embeddings for the remaining timesteps in reverse so as to continue the sinusoidal pattern
            for i in range(0, x.shape[1], self.max_len):
                x[:, i:i+self.max_len] = x[:, i:i+self.max_len] + self.pe[:x.shape[1]-i]
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
        #std = 1.0 * (2.0 / (1 + self.input_size))**0.5
        #self.infill_embeddings = nn.Parameter(torch.normal(0, std**2, (self.input_size,)))        
        # self.positional_encoding = PositionalEncoding(self.input_size, self.max_len)
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

        output_rep = embeddings #+ (infill_mask.unsqueeze(-1)*self.infill_embeddings)
        # adding positional encoding to the embeddings
        # output_rep = self.positional_encoding(embeddings)
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
    
class AutoRegressiveLinear(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, num_classes:int, num_outcomes:int=1, num_layers:int=1, 
                output_dropout:float=0.0, max_len:int=1):
        super(AutoRegressiveLinear, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_dropout = output_dropout
        self.num_classes = num_classes
        self.num_outcomes = num_outcomes
        self.max_len = max_len
        self.padding_len = self.max_len - 1 #Padding length for causal convolution to make the output of the same length as the input
        self.init_model()
        print (self)
        
    def init_model(self):
        self.model = []
        self.output_dropout_layer = nn.Dropout(self.output_dropout)        
        # for _ in range(self.num_layers-1):
        #     self.model.append(nn.Conv1d(in_channels=self.input_size, out_channels=self.hidden_size, kernel_size=self.max_len, bias=True))
        
        if self.num_classes<=2: # For binary classification or regression, we need only one output node
            self.model.append(nn.Conv1d(in_channels=self.input_size, out_channels=1*self.num_outcomes, kernel_size=self.max_len, bias=True))
        else:
            self.model.append(nn.Conv1d(in_channels=self.input_size, out_channels=self.num_classes*self.num_outcomes, kernel_size=self.max_len, bias=True))

        # Initial weights to 0
        self.model[0].weight.data.fill_(1)
        self.model[0].bias.data.fill_(0)
        self.model = nn.ModuleList(self.model)
        
    def forward(self, embeddings, mask, predict_last_valid_hidden_state=True, **kwargs):
        """
            embeddings: (batch_size, seq_len, input_size)
            mask: (batch_size, seq_len) of type bool. if mask = 1, then the query is was padded and should not be used for attention. If mask = 0, then the query is valid and should be used for attention.
            predict_last_valid_hidden_state: If True, then the last valid timestep's hidden state from each instance is used for prediction. Else, all timesteps' hidden states are predicted.
        """
        # x shape: (batch_size, seq_len, features)
        # Assuming features==1 here. If more, you'll need to adjust in_channels.
        # Permute to (batch_size, features, seq_len) for Conv1D.
        embeddings = embeddings.permute(0, 2, 1)
        if self.padding_len > 0:
            # Pad on the left for causal convolution with a tensor of input dimensions: (batch_size, features, padding_len)
            embeddings = nn.functional.pad(embeddings, (self.padding_len, 0))
        # Apply convolution. The output shape will depend on your padding/stride.
        model_out = self.model[0](embeddings)
        # Permute back to (batch_size, seq_len', 1)
        model_out_perm = model_out.permute(0, 2, 1)
        # Apply dropout
        model_out_perm_do = self.output_dropout_layer(model_out_perm)
        # import pdb; pdb.set_trace()
        
        # TODO: Include logic for predict_last_valid_hidden_state
        return model_out_perm_do
    
#################################


class AutoRegressiveLinear2(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int, num_outcomes: int = 1, 
                 output_dropout: float = 0.0, max_len: int = 1, **kwargs):
        super(AutoRegressiveLinear2, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_outcomes = num_outcomes
        self.max_len = max_len
        self.output_dropout = output_dropout
        self.padding_len = self.max_len - 1  # Padding length for causal sliding
        self.init_model()
        print (self)

    def init_model(self):
        # Define the linear layer
        if self.num_classes <= 2:  # Binary classification or regression
            self.linear = nn.Linear(self.input_size * self.max_len, 1 * self.num_outcomes, bias=True)
        else:  # Multi-class classification
            self.linear = nn.Linear(self.input_size * self.max_len, self.num_classes * self.num_outcomes, bias=True)
        
        # Dropout layer
        self.output_dropout_layer = nn.Dropout(self.output_dropout)
        
        # Initialize weights
        # self.linear.weight.data.fill_(1)
        # self.linear.bias.data.fill_(0)

    def forward(self, embeddings, mask=None, predict_last_valid_hidden_state=True, **kwargs):
        """
        embeddings: (batch_size, seq_len, input_size)
        mask: (batch_size, seq_len) of type bool. If mask = 1, the query was padded and should not be used for attention. If mask = 0, the query is valid and should be used for attention.
        predict_last_valid_hidden_state: If True, the last valid timestep's hidden state from each instance is used for prediction. Else, all timesteps' hidden states are predicted.
        """
        batch_size, seq_len, input_size = embeddings.shape
        assert input_size == self.input_size, "Input size mismatch. Expected {}, got {}".format(self.input_size, input_size)
        
        # Pad the input on the left for causal sliding
        if self.padding_len > 0:
            embeddings = nn.functional.pad(embeddings, (0, 0, self.padding_len, 0))  # Pad along the sequence length dimension
        
        # Create sliding windows of size `max_len`
        # Unfold the input tensor to create overlapping windows
        unfolded = embeddings.unfold(dimension=1, size=self.max_len, step=1)  # Shape: (batch_size, seq_len, max_len, input_size)
        unfolded = unfolded.contiguous().view(batch_size, seq_len, -1)  # Shape: (batch_size, seq_len, max_len * input_size)
        
        # Apply the linear layer to each window
        output = self.linear(unfolded)  # Shape: (batch_size, seq_len, num_classes * num_outcomes)
        
        # Apply dropout
        output = self.output_dropout_layer(output)
        
        # if mask is not None:
        #     # Zero out invalid timesteps using the mask
        #     output = output * (~mask).unsqueeze(-1)
        
        # if predict_last_valid_hidden_state:
        #     # Extract the last valid timestep's hidden state
        #     idx = torch.sum(~mask, dim=1) - 1  # Find the last valid timestep
        #     output = output[torch.arange(batch_size), idx]  # Shape: (batch_size, num_classes * num_outcomes)
        
        return output
