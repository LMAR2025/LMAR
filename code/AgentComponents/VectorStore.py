import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages import *
from utils import mean_pooling
from utils import (root_path, library_path, model_output_path, vectorstore_path)

class VectorStore:
    def __init__(self, model = None, tokenizer = None, database_path = None, train=False, write=False, device='cpu',max_length = 2000):
        self.train = train
        self.device = device

        if train:
            self.write = False
        else:
            self.write = write

        # set tokenizer config
        self.max_len = max_length
        
        if model:
            self.model = model
        else:
            best_model_path = model_output_path.joinpath(f"best_model")
            assert os.path.exists(best_model_path) and os.path.isdir(best_model_path), "There is no default model."
            
            self.model = AutoModel.from_pretrained(best_model_path)

        try:
            self.model_dim = self.model.config.hidden_size
        except AttributeError:
            try:
                self.model_dim = self.model.config.dim
            except AttributeError:
                raise ValueError("Unable to recognize hidden size from model config, please set model_dim in VectorStore manually.")


        if self.model.device.type != 'cuda':
            self.model = self.model.to(self.device)
            
        if tokenizer:
            self.tokenizer = tokenizer

        else:
            best_model_path = model_output_path.joinpath(f"best_model")
            assert os.path.exists(best_model_path) and os.path.isdir(best_model_path), "There is no default tokenizer."
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_output_path.joinpath(f"best_model"))

        if database_path:
            self.database_path = Path(database_path) if isinstance(database_path, str) else database_path
            if not(os.path.exists(database_path) and os.path.isdir(database_path)):
                os.makedirs(database_path)
            
            self.database_path = database_path
            self.vectorstore_path = self.database_path.joinpath('VectorStore_library.pkl')
            self.embedding_path = self.database_path.joinpath('embeddings.pth')

            if os.path.exists(self.vectorstore_path) and not self.write:
                with open(self.vectorstore_path, 'rb') as file:
                    self.database = pickle.load(file)
            else:
                self.database = {}

        else:
            root_path = Path('.').resolve().parent
            self.database_path = root_path.joinpath('VectorStore/')
            self.vectorstore_path = self.database_path.joinpath('VectorStore_library.pkl')
            self.embedding_path = self.database_path.joinpath('embeddings.pth')

            if os.path.exists(self.vectorstore_path) and not self.write:
                with open(self.vectorstore_path, 'rb') as file:
                    self.database = pickle.load(file)
            else:
                self.database = {}
            
        self.number_of_data = len(self.database)

        if os.path.exists(self.embedding_path) and not self.train:
            self.embedding = torch.load(self.embedding_path)
        else:
            self.embedding = torch.zeros(len(self.database), self.model_dim)
    
    def load_new_text(self, inputs = None, save=True):
        if self.train:
            save = False

        for doc in inputs:
            new_doc = {}
            new_doc['source'] = doc.metadata['source']
            new_doc['text'] = doc.page_content
            new_doc['embedding'] = None

            self.database[self.number_of_data] = new_doc
            
            self.number_of_data += 1

        if save is True:
            with open(self.vectorstore_path, 'wb') as file:
                pickle.dump(self.database, file)

    def embed_library(self, batch_size=128):

        unembedded_indices = [idx for idx, doc in self.database.items() if doc['embedding'] is None]
        unembedded_texts = [self.database[idx]['text'] for idx in unembedded_indices]

        if not unembedded_texts:  
            return

        new_embeddings = []  

        for i in tqdm(range(0, len(unembedded_texts), batch_size), desc="Embedding texts", unit="batch"):
            batch_texts = unembedded_texts[i:i + batch_size]

            # Tokenization
            encoded_input = self.tokenizer(batch_texts, padding=True, truncation=True,
                                           return_tensors='pt', max_length=self.max_len)
            encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

            with torch.no_grad():
                model_output = self.model(**encoded_input)
                batch_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
                batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)

            new_embeddings.append(batch_embeddings.cpu())  

        new_embeddings = torch.cat(new_embeddings, dim=0)  

        if self.embedding.shape[0] < len(self.database):
            pad_size = len(self.database) - self.embedding.shape[0]
            pad = torch.zeros((pad_size, self.model_dim))
            self.embedding = torch.cat((self.embedding, pad), dim=0)

        for idx, emb in zip(unembedded_indices, new_embeddings):
            self.database[idx]['embedding'] = emb
            self.embedding[idx] = emb

        self.embedding = self.embedding.to(self.device)

        if not self.train:
            torch.save(self.embedding, self.embedding_path)


    def retrieve(self, query = None, top_k = 5):
        # Get embeddings for input query
        encoded_input = self.tokenizer(query, padding='max_length', truncation=True, return_tensors='pt', max_length = self.max_len)
        encoded_input = {k: v.to(self.device) for k, v in encoded_input.items()}

        model_output = self.model(**encoded_input)
        query_embedding = mean_pooling(model_output, encoded_input['attention_mask'])

        query_norm = query_embedding / torch.norm(query_embedding, p=2, dim=1, keepdim=True)

        cosine_similarities = torch.mm(query_norm, self.embedding.t()).squeeze(0)

        top_k_similarities, top_k_indices = torch.topk(cosine_similarities, k=top_k, largest=True)

        retrieved_text = []
        for i in top_k_indices:
            retrieved_text.append(self.database[int(i)])
        
        return top_k_similarities, retrieved_text
