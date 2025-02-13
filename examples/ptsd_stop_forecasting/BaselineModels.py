import torch
import torch.nn as nn

# NOTE: Add the class to the BSLN_ARCHS dictionary at the bottom of the file.

class LinearRegression(nn.Module):
    """ Simple linear regression model. Use L2 penalty for regularization to use it as Ridge regression. """
    def __init__(self, input_size, output_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, embeddings_lang, mask_lang, **kwargs):
        return self.linear(embeddings_lang)
    
    
class LinearRegressionWithLN(nn.Module):
    """ Linear regression model with layer normalization before the linear layer. """
    def __init__(self, input_size, output_size, elementwise_affine=True):
        """
        Args:
            input_size: int, size of the input features
            output_size: int, size of the output features
            elementwise_affine: bool, whether to learn the affine parameters for layer normalization. Default is True.
        """
        super(LinearRegressionWithLN, self).__init__()
        self.layernorm = nn.LayerNorm(input_size, elementwise_affine=elementwise_affine)
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, embeddings_lang, mask_lang, **kwargs):
        x = self.layernorm(embeddings_lang)
        return self.linear(x) 
    

##############################################
    
BSLN_ARCHS = {
    'linear': LinearRegression,
    'linear_ln': LinearRegressionWithLN
}