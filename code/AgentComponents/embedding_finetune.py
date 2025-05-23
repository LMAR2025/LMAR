import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages import *
from utils import split_and_filter_paragraph

from sentence_transformers import SentenceTransformer, InputExample, losses, models, datasets
from torch.utils.data import DataLoader, Dataset
from sklearn.feature_extraction.text import CountVectorizer


from AgentComponents.VectorStore import VectorStore


class MyDataset(Dataset):
    def __init__(self, examples, tokenizer, max_len=1500):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        texts = example.texts
        label = example.label

        # Tokenize both sentences
        encoded_input = self.tokenizer(texts, padding='max_length', truncation=True, return_tensors='pt',
                                       max_length=self.max_len)

        return encoded_input, torch.tensor(label, dtype=torch.float)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def split_batch(batch_data):
    # Access the input_ids and attention_mask from the batch_data dictionary
    input_ids = batch_data['input_ids']  # Shape: [batch, 2, length]
    attention_mask = batch_data['attention_mask']  # Shape: [batch, 2, length]

    # Split input_ids and attention_mask along the second dimension
    input_ids_question = input_ids[:, 0, :]  # First part (question) of shape [batch, length]
    input_ids_context = input_ids[:, 1, :]   # Second part (context) of shape [batch, length]

    attention_mask_question = attention_mask[:, 0, :]  # First part (question)
    attention_mask_context = attention_mask[:, 1, :]   # Second part (context)

    batch_question = {'input_ids':input_ids_question, 'attention_mask': attention_mask_question}
    batch_context = {'input_ids':input_ids_context, 'attention_mask': attention_mask_context}

    return batch_question, batch_context

def tfidf_overlap_similarity(text_a, text_b):
    TFvectorizer = CountVectorizer()
    tfidf_matrix = TFvectorizer.fit_transform([text_a, text_b])
    tfidf_a = tfidf_matrix[0].toarray()
    tfidf_b = tfidf_matrix[1].toarray()

    matching_score = (tfidf_a * tfidf_b).sum() / tfidf_b.sum() if tfidf_b.sum() else 0
    return matching_score


def get_random_content(combined_content, retrieved_content):
    while True:
        random_content = random.choice(combined_content)
        if random_content not in retrieved_content:
            return random_content


def create_dataset(df_train, tokenizer, max_len, pos_neg_ratio=4):  # pos_neg_ratio=4
    train_examples = []

    combined_content = split_and_filter_paragraph(' \n\n '.join(df_train['Evidence'].dropna()))
    combined_content = list(set(combined_content))

    for idx, row in tqdm(df_train.iterrows(), total=len(df_train), desc="Create Dataset"):
        question = row['Question']
        retrieved_content = split_and_filter_paragraph(
            row['Evidence'])  # Todo: retrieved_paragraph_content split to sentence
        for retrieved_single in retrieved_content:
            # Create Possitive InputExamples
            train_examples.append(InputExample(texts=[question, retrieved_single], label=1))
            # Create Negative InputExamples - random (2:8)
            for _ in range(pos_neg_ratio):
                random_content = get_random_content(combined_content, retrieved_content)
                train_examples.append(InputExample(texts=[question, random_content], label=-1))

    dataset = MyDataset(train_examples, tokenizer, max_len=max_len)
    # dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataset
    # return dataloader
    # return train_examples


def gamma_pdf(x, alpha, theta):
    alpha = torch.tensor([alpha], dtype = torch.float64)
    # Compute the gamma function Γ(alpha)
    gamma_alpha = torch._standard_gamma(alpha)
    
    # Calculate each component of the gamma PDF
    x_power = torch.pow(x, alpha - 1)
    exp_term = torch.exp(-x / theta)
    denominator = (theta ** alpha) * gamma_alpha
    
    # Combine components to compute the PDF
    pdf = (x_power * exp_term) / denominator
    
    return pdf

def gamma_pdf(x, alpha, theta):
    alpha = torch.tensor([alpha], dtype = torch.float64)
    # Compute the gamma function Γ(alpha)
    gamma_alpha = torch._standard_gamma(alpha)
    
    # Calculate each component of the gamma PDF
    x_power = torch.pow(x, alpha - 1)
    exp_term = torch.exp(-x / theta)
    denominator = (theta ** alpha) * gamma_alpha
    
    # Combine components to compute the PDF
    pdf = (x_power * exp_term) / denominator
    
    return pdf

def gamma_cdf(x, alpha, theta):
    alpha = torch.tensor([alpha], dtype = torch.float64).to(x.device)
    input = alpha
    other = x/theta
    cdf = torch.special.gammainc(input = input, other = other)
    return cdf

def positive_loss(x, alpha, theta):
    loss = gamma_cdf((1-x)*20,alpha = alpha, theta=theta)
    return loss

def negative_loss(x, alpha, theta):
    margin = 0.65
    mask = x >= margin
    loss = gamma_cdf((x)*3, alpha = alpha, theta = theta)
    loss = loss * mask
    return loss

def neutral_loss(x, loc = 0.85):
    loss = (3*(x-loc))**2
    bounded_loss = loss / (1 + loss)
    return bounded_loss

def combined_loss(x, label):
    neg_loss = negative_loss(x, alpha = 0.5, theta = 1)
    neg_mask = label == -1
    neg_loss = neg_loss * neg_mask

    mid_loss = neutral_loss(x,loc=0.7)
    mid_mask = label == 0
    mid_loss = mid_loss * mid_mask

    pos_loss = positive_loss(x, alpha = 0.5, theta = 1)
    pos_mask = label == 1
    pos_loss = pos_loss * pos_mask

    return torch.mean(neg_loss) + torch.mean(mid_loss) + torch.mean(pos_loss)

def scaled_similarity_loss(embeddings1, embeddings2, labels):
    embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
    embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)
    cosine_similarities = torch.sum(embeddings1*embeddings2,dim = 1)
    cosine_similarities = cosine_similarities*0.5+0.5
    loss = combined_loss(cosine_similarities, labels)
    return loss