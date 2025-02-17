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
    

class LinearRegressionSubscales(nn.Module):
    """ Simple linear regression model for subscales. """
    def __init__(self, input_size, output_size):
        super(LinearRegressionSubscales, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        return self.linear(embeddings_subscales)


class LinearRegressionSubscalesBN(nn.Module):
    """ Linear Regression with Batch Norm before the linear layer"""
    def __init__(self, input_size, output_size, affine=True):
        """
        Args:
            input_size: int, size of the input features
            output_size: int, size of the output features
            affine: bool, whether to learn the affine parameters for batch normalization. Default is True.
        """
        super(LinearRegressionSubscalesBN, self).__init__()
        self.bn = nn.BatchNorm1d(input_size, affine=affine)
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        # Perform transpose to bring the feature dimensions from last to second before applying batch norm
        x = self.bn(embeddings_subscales.transpose(1, 2)).transpose(1, 2)
        return self.linear(x)
    

class LinearRegressionSubscalesZ(nn.Module):
    """ Simple Linear Regression on Z-scores Subscales """
    def __init__(self, input_size, output_size):
        super(LinearRegressionSubscalesZ, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        
    def forward(self, embeddings_subscales_z, mask_subscales, **kwargs):
        return self.linear(embeddings_subscales_z) 

##############################################
    
BSLN_ARCHS = {
    'linear': LinearRegression,
    'linear_ln': LinearRegressionWithLN,
    'linear_subscales': LinearRegressionSubscales,
    'linear_subscales_bn': LinearRegressionSubscalesBN,
    'linear_subscales_z': LinearRegressionSubscalesZ
}