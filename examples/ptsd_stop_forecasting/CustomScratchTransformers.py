from utils import add_to_path
add_to_path(__file__)

import torch.nn as nn
# from torch.amp import autocast
import torch

from src import TransformerModel

class DailyLangZFormer(nn.Module):
    """
        Transformer that uses daily language embeddings to predict the PCL score.
    """
    def __init__(self, lang_transformer:TransformerModel):
        super(DailyLangZFormer, self).__init__()
        self.lang_transformer = lang_transformer
    
    # @autocast(device_type='cuda')
    def forward(self, embeddings_lang_z, mask_lang, **kwargs):
        output = self.lang_transformer(embeddings=embeddings_lang_z, mask=mask_lang, **kwargs)
        return output
    
SCTRNS_ARCH = {
    'dailylangzformer': DailyLangZFormer,
}