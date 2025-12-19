from utils import add_to_path
add_to_path(__file__)

import torch.nn as nn
import torch

from src import AutoRegressiveLinear, AutoRegressiveLinear2, BoELinear

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


class LastNPCLMean(nn.Module):
    """ Last N days' mean value of the PCL """
    def __init__(self, subscaleAR: AutoRegressiveLinear2):
        super(LastNPCLMean, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.max_len = self.subscaleAR_model.max_len
        self.subscaleAR_model.linear.weight.data.fill_(1)
        self.subscaleAR_model.linear.bias.data.fill_(0)

        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        # with torch.no_grad():
        output = self.subscaleAR_model(embeddings_subscales[:, :, :1], mask_subscales, **kwargs)
        if self.max_len > 1:
            M = min(self.max_len, embeddings_subscales.shape[1])
            # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
            divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_subscales.device), 
                                        torch.full((1, embeddings_subscales.shape[1] - M + 1), M).to(embeddings_subscales.device)], 
                                    dim=1).unsqueeze(-1)
            # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales.device).expand(1, embeddings_subscales.shape[1], 1)
            # Shape: [1, seq_len, 1]
            output = output / divy_tensor.to(torch.float32)
        
        # output.requires_grad = True
        return output


class BoELangZ(nn.Module):
    """ Bag of Embeddings Language Model with Z-scores """
    def __init__(self, model:BoELinear):
        super(BoELangZ, self).__init__()
        self.model = model
    
    def forward(self, embeddings_lang_z, mask_lang, **kwargs):
        output = self.model(embeddings_lang_z, mask_lang, **kwargs)
        return output
        

class ARSubscale(nn.Module):
    """Auto Regressive Subscale Model with given history length"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2):
        super(ARSubscale, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.max_len = self.subscaleAR_model.max_len
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        output = self.subscaleAR_model(embeddings_subscales, mask_subscales, **kwargs)
        if self.max_len > 1:
            M = min(self.max_len, embeddings_subscales.shape[1])
            # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
            divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_subscales.device), 
                                        torch.full((1, embeddings_subscales.shape[1] - M + 1), M).to(embeddings_subscales.device)], 
                                    dim=1).unsqueeze(-1)
            # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales.device).expand(1, embeddings_subscales.shape[1], 1)
            # Shape: [1, seq_len, 1]
            output = output / divy_tensor.to(torch.float32)
        return output


class ARSubscaleZ(nn.Module):
    """Auto Regressive Subscale Model with Z-scores"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2):
        super(ARSubscaleZ, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.max_len = self.subscaleAR_model.max_len
        
    def forward(self, embeddings_subscales_z, mask_subscales, **kwargs):
        output = self.subscaleAR_model(embeddings_subscales_z, mask_subscales, **kwargs)
        # if self.max_len > 1:
        #     M = min(self.max_len, embeddings_subscales_z.shape[1])
        #     # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
        #     divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_subscales_z.device), 
        #                                 torch.full((1, embeddings_subscales_z.shape[1] - M + 1), M).to(embeddings_subscales_z.device)], 
        #                             dim=1).unsqueeze(-1)
        #     # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales_z.device).expand(1, embeddings_subscales_z.shape[1], 1)
        #     # Shape: [1, seq_len, 1]
        #     output = output / divy_tensor.to(torch.float32)
        return output


class ARPCLZLangZ(nn.Module):
    """Auto Regressive PCL total Model with Z-scores and Language concatenated"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2, langAR:AutoRegressiveLinear2):
        super(ARPCLZLangZ, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.langAR_model = langAR
        self.max_len = self.subscaleAR_model.max_len
        self.input_dropout = nn.Dropout(0.1)
        
    def forward(self, embeddings_subscales_z, mask_subscales, embeddings_lang_z, mask_lang, **kwargs):
        with torch.no_grad():
            output_subscale = self.subscaleAR_model(embeddings_subscales_z[:, :, :1], mask_subscales, **kwargs)
        embeddings_lang_z_do = self.input_dropout(embeddings_lang_z) 
        output_lang = self.langAR_model(embeddings_lang_z_do, mask_lang, **kwargs)
        output = output_subscale + output_lang
        return output


class ARPCLZ(nn.Module):
    """Auto Regressive PCL total Model with Z-scores"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2):
        super(ARPCLZ, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.max_len = self.subscaleAR_model.max_len
        
    def forward(self, embeddings_subscales_z, mask_subscales, **kwargs):
        output = self.subscaleAR_model(embeddings_subscales_z[:, : ,:1], mask_subscales, **kwargs)
        # if self.max_len > 1:
        #     M = min(self.max_len, embeddings_subscales_z.shape[1])
        #     # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
        #     divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_subscales_z.device), 
        #                                 torch.full((1, embeddings_subscales_z.shape[1] - M + 1), M).to(embeddings_subscales_z.device)], 
        #                             dim=1).unsqueeze(-1)
        #     # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales_z.device).expand(1, embeddings_subscales_z.shape[1], 1)
        #     # Shape: [1, seq_len, 1]
        #     output = output / divy_tensor.to(torch.float32)
        return output


class ARPCLZMissingIndicator(nn.Module):
    """Auto Regressive PCL total Model with Z-scores and Masks as Missing Indicator"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2):
        super(ARPCLZMissingIndicator, self).__init__()
        self.subscaleAR_model = subscaleAR
        
    def forward(self, embeddings_subscales_z, mask_subscales, **kwargs):
        input_tensor = torch.cat([embeddings_subscales_z[:, :, :1], 1 - mask_subscales.unsqueeze(-1).to(embeddings_subscales_z.dtype)], dim=2)
        output = self.subscaleAR_model(input_tensor, mask_subscales, **kwargs)
        return output


class ARPCL(nn.Module):
    """Auto Regressive PCL total Model"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2):
        super(ARPCL, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.max_len = self.subscaleAR_model.max_len
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        output = self.subscaleAR_model(embeddings_subscales[:, : ,:1], mask_subscales, **kwargs)
        if self.max_len > 1:
            M = min(self.max_len, embeddings_subscales.shape[1])
            # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
            divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_subscales.device), 
                                        torch.full((1, embeddings_subscales.shape[1] - M + 1), M).to(embeddings_subscales.device)], 
                                    dim=1).unsqueeze(-1)
            # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales.device).expand(1, embeddings_subscales.shape[1], 1)
            # Shape: [1, seq_len, 1]
            output = output / divy_tensor.to(torch.float32)
        return output


class ARSubscaleMissingIndicator(nn.Module):
    """Auto Regressive Subscale Model with Z-scores and Masks as Missing Indicator"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2):
        super(ARSubscaleMissingIndicator, self).__init__()
        self.subscaleAR_model = subscaleAR
        
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        input_tensor = torch.cat([embeddings_subscales, 1 - mask_subscales.unsqueeze(-1).to(embeddings_subscales.dtype)], dim=2)
        output = self.subscaleAR_model(input_tensor, mask_subscales, **kwargs)
        if self.subscaleAR_model.max_len > 1:
            M = min(self.subscaleAR_model.max_len, embeddings_subscales.shape[1])
            # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
            divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_subscales.device), 
                                        torch.full((1, embeddings_subscales.shape[1] - M + 1), M).to(embeddings_subscales.device)], 
                                    dim=1).unsqueeze(-1)
            # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales.device).expand(1, embeddings_subscales.shape[1], 1)
            # Shape: [1, seq_len, 1]
            output = output / divy_tensor.to(torch.float32)
        return output
    
    
class ARSubscaleMissingIndicatorLangMissingIndicator(nn.Module):
    """Auto Regressive Subscale Model with Z-scores and Masks as Missing Indicator"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2, langAR:AutoRegressiveLinear2):
        super(ARSubscaleMissingIndicatorLangMissingIndicator, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.langAR_model = langAR
        self.max_len = self.subscaleAR_model.max_len
        
    def forward(self, embeddings_subscales, mask_subscales, embeddings_lang, mask_lang, **kwargs):
        with torch.no_grad():
            subscale_input_tensor = torch.cat([embeddings_subscales, 1 - mask_subscales.unsqueeze(-1).to(embeddings_subscales.dtype)], dim=2)
            subscale_output = self.subscaleAR_model(subscale_input_tensor, mask_subscales, **kwargs)
            if self.subscaleAR_model.max_len > 1:
                M = min(self.subscaleAR_model.max_len, embeddings_subscales.shape[1])
                # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
                divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_subscales.device), 
                                            torch.full((1, embeddings_subscales.shape[1] - M + 1), M).to(embeddings_subscales.device)], 
                                        dim=1).unsqueeze(-1)
                # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales.device).expand(1, embeddings_subscales.shape[1], 1)
                # Shape: [1, seq_len, 1]
                subscale_output = subscale_output / divy_tensor.to(torch.float32)
        
        lang_input_tensor = torch.cat([embeddings_lang, 1 - mask_lang.unsqueeze(-1).to(embeddings_lang.dtype)], dim=2)
        lang_output = self.langAR_model(lang_input_tensor, mask_lang, **kwargs)
        if self.langAR_model.max_len > 1:
            M = min(self.langAR_model.max_len, embeddings_lang.shape[1])
            # The output should be averaged by the number of days (max_len) for the last N - max_len + 1 days. For the first max_len - 1 days, the outputs should be divided by rage(1, max_len)
            divy_tensor = torch.cat([torch.arange(1, M).unsqueeze(0).to(embeddings_lang.device), 
                                        torch.full((1, embeddings_lang.shape[1] - M + 1), M).to(embeddings_lang.device)], 
                                    dim=1).unsqueeze(-1)
            # divy_tensor = torch.tensor(self.max_len).to(embeddings_subscales.device).expand(1, embeddings_subscales.shape[1], 1)
            # Shape: [1, seq_len, 1]
            lang_output = lang_output / divy_tensor.to(torch.float32)
        output = subscale_output + lang_output
        return output
    

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


class ARSubscaleLangCat(nn.Module):
    """ Auto Regressive Subscale with Language concatenated """
    def __init__(self, ARmodel:AutoRegressiveLinear):
        super(ARSubscaleLangCat, self).__init__()
        self.ARmodel = ARmodel
        
    def forward(self, embeddings_subscales, mask_subscales, embeddings_lang, mask_lang, **kwargs):
        embeddings = torch.cat([embeddings_subscales, embeddings_lang], dim=2)
        # Take the logical OR for the masks
        mask = mask_subscales | mask_lang
        return self.ARmodel(embeddings, mask, **kwargs)
    
    
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


class ARSubscaleZLangZ(nn.Module):
    """Auto Regressive Subscale with Language concatenated"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2, LangAR:AutoRegressiveLinear2):
        super(ARSubscaleZLangZ, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.langAR_model = LangAR
        self.input_dropout = nn.Dropout(0.1)
        # Init langAR model with 0 weights
        # self.langAR_model.linear.weight.data.fill_(0.0)
        # self.langAR_model.linear.bias.data.fill_(0.0)
        
    def forward(self, embeddings_subscales_z, mask_subscales, embeddings_lang_z, mask_lang, **kwargs):
        with torch.no_grad():
            output_subscale = self.subscaleAR_model(embeddings_subscales_z, mask_subscales, **kwargs)
        embeddings_lang_z_do = self.input_dropout(embeddings_lang_z) 
        output_lang = self.langAR_model(embeddings_lang_z_do, mask_lang, **kwargs)
        output = output_subscale + output_lang
        # return (output, (output_subscale, output_lang))
        return output

class ARSubscaleZLangZCat(nn.Module):
    """ Auto Regressive Subscale with Language concatenated """
    def __init__(self, ARmodel:AutoRegressiveLinear2):
        super(ARSubscaleZLangZCat, self).__init__()
        self.ARmodel = ARmodel
        self.input_dropout = nn.Dropout(0.1)
        
    def forward(self, embeddings_subscales_z, mask_subscales, embeddings_lang_z, mask_lang, **kwargs):
        embeddings = torch.cat([embeddings_subscales_z, embeddings_lang_z], dim=2)
        embeddings_do = self.input_dropout(embeddings)
        # Take the logical OR for the masks
        mask = mask_subscales | mask_lang
        return self.ARmodel(embeddings_do, mask, **kwargs)


class ARSubscaleZMissingIndicator(nn.Module):
    """Auto Regressive Subscale Model with Z-scores and Masks as Missing Indicator"""
    def __init__(self, subscaleAR:AutoRegressiveLinear2):
        super(ARSubscaleZMissingIndicator, self).__init__()
        self.subscaleAR_model = subscaleAR
        
    def forward(self, embeddings_subscales_z, mask_subscales, **kwargs):
        input_tensor = torch.cat([embeddings_subscales_z, 1 - mask_subscales.unsqueeze(-1).to(embeddings_subscales_z.dtype)], dim=2)
        return self.subscaleAR_model(input_tensor, mask_subscales, **kwargs)
    

class ARSubscaleZLangZMissingIndicators(nn.Module):
    """Auto Regressive Subscale Model with Z-scores and Masks as Missing Indicator"""
    def __init__(self, subscaleAR:AutoRegressiveLinear, langAR:AutoRegressiveLinear):
        super(ARSubscaleZLangZMissingIndicators, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.langAR_model = langAR
        
    def forward(self, embeddings_subscales_z, mask_subscales, embeddings_lang_z, mask_lang, **kwargs):
        input_tensor_subscale = torch.cat([embeddings_subscales_z, mask_subscales.unsqueeze(-1).to(embeddings_subscales_z.dtype)], dim=2)
        with torch.no_grad():
            output_subscale = self.subscaleAR_model(input_tensor_subscale, mask_subscales, **kwargs)
        input_tensor_lang = torch.cat([embeddings_lang_z, mask_lang.unsqueeze(-1).to(embeddings_lang_z.dtype)], dim=2)
        output_lang = self.langAR_model(input_tensor_lang, mask_lang, **kwargs)
        output = output_subscale + output_lang
        return output


class ARSubscaleZLastObserved(nn.Module):
    """Auto Regressive Subscale Model with Z-scores and continuous integral value denoting last observed value"""
    def __init__(self, subscaleAR:AutoRegressiveLinear):
        super(ARSubscaleZLastObserved, self).__init__()
        self.subscaleAR_model = subscaleAR
        
    def forward(self, embeddings_subscales_z, mask_subscales, embeddings_lang_z, mask_lang, **kwargs):
        lastobserved_subscales = kwargs["lastobserved_subscales"].unsqueeze(-1).to(embeddings_subscales_z.dtype)
        input_tensor_subscale = torch.cat([embeddings_subscales_z, lastobserved_subscales], dim=2)
        output = self.subscaleAR_model(input_tensor_subscale, mask_subscales, **kwargs)/lastobserved_subscales
        return output



class ARSubscaleZLangZLastObserved(nn.Module):
    """Auto Regressive Subscale Model with Z-scores and continuous integral value denoting last observed value"""
    def __init__(self, subscaleAR:AutoRegressiveLinear, langAR:AutoRegressiveLinear):
        super(ARSubscaleZLangZLastObserved, self).__init__()
        self.subscaleAR_model = subscaleAR
        self.langAR_model = langAR
        
    def forward(self, embeddings_subscales_z, mask_subscales, embeddings_lang_z, mask_lang, **kwargs):
        lastobserved_subscales = kwargs["lastobserved_subscales"].unsqueeze(-1).to(embeddings_subscales_z.dtype)
        input_tensor_subscale = torch.cat([embeddings_subscales_z, lastobserved_subscales], dim=2)
        with torch.no_grad():
            output_subscale = self.subscaleAR_model(input_tensor_subscale, mask_subscales, **kwargs)#/lastobserved_subscales
        lastobserved_lang = kwargs["lastobserved_lang"].unsqueeze(-1).to(embeddings_lang_z.dtype)
        input_tensor_lang = torch.cat([embeddings_lang_z, lastobserved_lang], dim=2)
        output_lang = self.langAR_model(input_tensor_lang, mask_lang, **kwargs)#/lastobserved_lang
        output = output_subscale + output_lang
        return output


class ARLangZ(nn.Module):
    """Auto Regressive Language Model with Z-scores"""
    def __init__(self, langAR:AutoRegressiveLinear2):
        super(ARLangZ, self).__init__()
        self.langAR_model = langAR
        self.max_len = self.langAR_model.max_len
        
    def forward(self, embeddings_lang_z, mask_lang, **kwargs):
        output = self.langAR_model(embeddings_lang_z, mask_lang, **kwargs)
        return output
    
##############################################
    
BSLN_ARCHS = {
    'linear': LinearRegression,
    'linear_ln': LinearRegressionWithLN,
    'linear_subscales': LinearRegressionSubscales,
    'linear_subscales_bn': LinearRegressionSubscalesBN,
    'linear_subscales_lang': LinearRegressionSubscalesLang,
    'linear_subscales_z': LinearRegressionSubscalesZ,
    'last_n_pcl_mean': LastNPCLMean,
    'boe_lang_z': BoELangZ,
    'ar_subscale': ARSubscale,
    'ar_subscale_z': ARSubscaleZ,
    'ar_pcl': ARPCL,
    'ar_subscale_missing': ARSubscaleMissingIndicator,
    'ar_subscale_missing_lang_missing': ARSubscaleMissingIndicatorLangMissingIndicator,
    'ar_pcl_z': ARPCLZ,
    'ar_pcl_z_lang_z': ARPCLZLangZ,
    'ar_pcl_z_missing': ARPCLZMissingIndicator,
    'ar_subscale_lang': ARSubscaleLang,
    'ar_subscale_lang_cat': ARSubscaleLangCat,
    'ar_subscale_z_lang': ARSubscaleZLang,
    'ar_subscale_z_lang_z': ARSubscaleZLangZ,
    'ar_subscale_z_lang_z_cat': ARSubscaleZLangZCat,
    'ar_subscale_z_missing': ARSubscaleZMissingIndicator,
    'ar_subscale_z_lang_z_missing': ARSubscaleZLangZMissingIndicators,
    'ar_subscale_z_lastobserved': ARSubscaleZLastObserved,
    'ar_subscale_z_lang_z_lastobserved': ARSubscaleZLangZLastObserved,
    'ar_lang_z': ARLangZ
}