import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages import *
from utils import access_token
from utils import mean_pooling

class EmbedderModule:
    def __init__(self, tokenizer = None, base_model = None, max_len = 512):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if tokenizer:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens')
        
        if base_model:
            self.model = base_model
        else:
            self.model = AutoModel.from_pretrained('sentence-transformers/distilbert-base-nli-stsb-mean-tokens').to(self.device)
            
        self.max_len = max_len
    
    def embed(self,texts):
        with torch.no_grad():
            encoded_input = self.tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt', max_length = 512).to(self.device)
            model_output = self.model(**encoded_input)
            embedding_vector = mean_pooling(model_output, encoded_input['attention_mask'])
            
        embedding_vector = embedding_vector.cpu()
        del encoded_input, model_output
        torch.cuda.empty_cache()
        return embedding_vector