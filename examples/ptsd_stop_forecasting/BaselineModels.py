from utils import add_to_path
add_to_path(__file__)

import torch.nn as nn
import torch

from src import AutoRegressiveLinear

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
        self.linear.weight.data.fill_(0.0)
        self.linear.bias.data.fill_(0.0)
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        return self.linear(embeddings_subscales)


class LinearRegressionSubscalesLang(nn.Module):
    """ Linear Regression model for Subscales with Language concatenated """
    def __init__(self, subscales_linear, lang_linear):
        super(LinearRegressionSubscalesLang, self).__init__()
        self.subscales_linear = subscales_linear
        self.lang_linear = lang_linear
    
    def forward(self, embeddings_subscales, mask_subscales, embeddings_lang, mask_lang, **kwargs):
        output = self.subscales_linear(embeddings_subscales, mask_subscales, **kwargs) + self.lang_linear(embeddings_lang, mask_lang, **kwargs)
        output /= 2.0
        return output


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


class ARSubscale(nn.Module):
    """Auto Regressive Subscale Model with given history length"""
    def __init__(self, subscaleAR:AutoRegressiveLinear):
        super(ARSubscale, self).__init__()
        self.subscaleAR_model = subscaleAR
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        return self.subscaleAR_model(embeddings_subscales, mask_subscales, **kwargs)


class ARSubscaleZ(nn.Module):
    """Auto Regressive Subscale Model with Z-scores"""
    def __init__(self, subscaleAR:AutoRegressiveLinear):
        super(ARSubscaleZ, self).__init__()
        self.subscaleAR_model = subscaleAR
        
    def forward(self, embeddings_subscales_z, mask_subscales, **kwargs):
        return self.subscaleAR_model(embeddings_subscales_z, mask_subscales, **kwargs)
    

class ARSubscaleShifted(nn.Module):
    """Auto Regressive Subscale Model with shifted embeddings"""
    def __init__(self, subscaleAR:AutoRegressiveLinear):
        super(ARSubscaleShifted, self).__init__()
        self.subscaleAR_model = subscaleAR
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        embeddings_subscales_shifted = torch.cat([torch.zeros_like(embeddings_subscales[:, :1]), embeddings_subscales[:, :-1]], dim=1)
        embeddings_subscales_diff = embeddings_subscales - embeddings_subscales_shifted
        diff_pred = self.subscaleAR_model(embeddings_subscales_diff, mask_subscales, **kwargs)
        pred = embeddings_subscales_shifted[:, :, :1] + diff_pred  
        # import pdb; pdb.set_trace()
        return pred

class ARSubscaleLang(nn.Module):
    """Auto Regressive Subscale with Language concatenated"""
    def __init__(self, subscaleAR:AutoRegressiveLinear, LangAR:AutoRegressiveLinear):
        super(ARSubscaleLang, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.langAR_model = LangAR
        
    def forward(self, embeddings_subscales, mask_subscales, embeddings_lang, mask_lang, **kwargs):
        output = self.subscaleAR_model(embeddings_subscales, mask_subscales, **kwargs) + self.langAR_model(embeddings_lang, mask_lang, **kwargs)
        # output /= 2.0
        return output
    
class ARSubscaleZLang(nn.Module):
    """Auto Regressive Subscale with Language concatenated"""
    def __init__(self, subscaleAR:AutoRegressiveLinear, LangAR:AutoRegressiveLinear):
        super(ARSubscaleZLang, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.langAR_model = LangAR
        
    def forward(self, embeddings_subscales_z, mask_subscales, embeddings_lang, mask_lang, **kwargs):
        output = self.subscaleAR_model(embeddings_subscales_z, mask_subscales, **kwargs) + self.langAR_model(embeddings_lang, mask_lang, **kwargs)
        # output /= 2.0
        return output
    
##############################################
    
BSLN_ARCHS = {
    'linear': LinearRegression,
    'linear_ln': LinearRegressionWithLN,
    'linear_subscales': LinearRegressionSubscales,
    'linear_subscales_bn': LinearRegressionSubscalesBN,
    'linear_subscales_lang': LinearRegressionSubscalesLang,
    'linear_subscales_z': LinearRegressionSubscalesZ,
    'ar_subscale': ARSubscale,
    'ar_subscale_z': ARSubscaleZ,
    'ar_subscale_shifted': ARSubscaleShifted,
    'ar_subscale_lang': ARSubscaleLang,
    'ar_subscale_z_lang': ARSubscaleZLang,
}