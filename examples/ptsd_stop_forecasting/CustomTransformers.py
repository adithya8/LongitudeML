from utils import add_to_path
add_to_path(__file__)

import torch.nn as nn
import torch

from src import AutoRegressiveTransformer, PositionalEncoding

class TotalPCLFormer(nn.Module):
    """
        Averages the uiinput subscale to form total PCL score before feeding it to the transformer.
    """
    def __init__(self, pcl_transformer:AutoRegressiveTransformer):
        super(TotalPCLFormer, self).__init__()
        self.pcl_transformer = pcl_transformer
    
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        embeddings_subscales = torch.mean(embeddings_subscales, dim=-1, keepdim=True)
        output = self.pcl_transformer(embeddings=embeddings_subscales, mask=mask_subscales, **kwargs)
        return output


class PCLSubscaleFormer(nn.Module):
    """
        Transformer that uses subscale embeddings to predict the PCL score.
    """
    def __init__(self, pcl_transformer:AutoRegressiveTransformer):
        super(PCLSubscaleFormer, self).__init__()
        self.pcl_transformer = pcl_transformer
    
    def forward(self, embeddings_subscales, mask_subscales, **kwargs):
        output = self.pcl_transformer(embeddings=embeddings_subscales, mask=mask_subscales, **kwargs)
        return output


class WTCPCLSubscaleFormer(nn.Module):
    """
        Transformer that uses subscale embeddings derived from WTC model to predict the PCL score.
    """
    def __init__(self, pcl_transformer:AutoRegressiveTransformer):
        super(WTCPCLSubscaleFormer, self).__init__()
        self.pcl_transformer = pcl_transformer
    
    def forward(self, embeddings_wtcSubscales, mask_wtcSubscales, **kwargs):
        output = self.pcl_transformer(embeddings=embeddings_wtcSubscales, mask=mask_wtcSubscales, **kwargs)
        return output
    

class DailyLangFormer(nn.Module):
    """
        Transformer that uses daily language embeddings to predict the PCL score.
    """
    def __init__(self, lang_transformer:AutoRegressiveTransformer):
        super(DailyLangFormer, self).__init__()
        self.lang_transformer = lang_transformer
    
    def forward(self, embeddings_lang, mask_lang, **kwargs):
        output = self.lang_transformer(embeddings=embeddings_lang, mask=mask_lang, **kwargs)
        return output
    

class LangSubscaleDualContextFormer(nn.Module):
    """
        Transformer that combines the output of two transformers, one for language and one for subscales.
        Averages the output of the two transformers to produce the final output.
    """
    def __init__(self, lang_transformer:AutoRegressiveTransformer, subscales_transformer:AutoRegressiveTransformer):
        super(LangSubscaleDualContextFormer, self).__init__()
        self.lang_transformer = lang_transformer
        self.subscales_transformer = subscales_transformer
        self.relu = nn.ReLU()
    
    def forward(self, embeddings_lang, mask_lang, embeddings_subscales, mask_subscales, **kwargs):
        lang_out = self.lang_transformer(embeddings_lang, mask_lang, **kwargs)
        subscales_out = self.subscales_transformer(embeddings_subscales, mask_subscales, **kwargs)
        lang_out = self.relu(lang_out)
        subscales_out = self.relu(subscales_out)
        output = (lang_out + subscales_out)/2.0
        # output = (lang_out + subscales_out)/2.0
        return output


class LexTransformer(nn.Module):
    """
        Transformer model for lexically derived features. 
    """
    def __init__(self, lex_transformer:AutoRegressiveTransformer):
        super(LexTransformer, self).__init__()
        # self.batchnorm = nn.BatchNorm1d(lex_transformer.input_size)
        self.lex_transformer = lex_transformer
        # self.relu = nn.ReLU()
    
    def forward(self, embeddings_hypLex, mask_hypLex, **kwargs):
        # embeddings = self.batchnorm(embeddings.transpose(1, 2)).transpose(1, 2)
        output = self.lex_transformer(embeddings=embeddings_hypLex, mask=mask_hypLex, **kwargs)
        # output = self.relu(output)
        return output

class projectedSubscaleTransformer(nn.Module):
    def __init__(self,subscalesformer:AutoRegressiveTransformer):
        super(projectedSubscaleTransformer,self).__init__()
        self.subscalesformer = subscalesformer
        self.layernorm = nn.LayerNorm(4,elementwise_affine=False)
        self.Linear = nn.Linear(in_features=5,out_features=4)

    def forward(self,embeddings,mask,**kwargs):
        output_rep = self.Linear(embeddings)
        output_rep = self.layernorm(output_rep)
        output = self.subscalesformer(embeddings=output_rep,mask=mask,**kwargs)
        return output

TRNS_ARCHS = {
    "totalpclformer": TotalPCLFormer,
    "wtcpclsubscaleformer": WTCPCLSubscaleFormer,
    "pclsubscaleformer": PCLSubscaleFormer,
    "dailylangformer": DailyLangFormer,
    "langsubscaledualcontextformer": LangSubscaleDualContextFormer,
    "lextransformer": LexTransformer,
    "projectedsubscaleformer":projectedSubscaleTransformer
}
