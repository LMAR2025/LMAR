from packages import *

root_path = Path.cwd().parent.resolve()

library_path = root_path.joinpath("./library")
output_path = root_path.joinpath("./outputs")
table_path = root_path.joinpath("table/")
vectorstore_path = root_path.joinpath("./VectorStore")
model_output_path = root_path.joinpath('embedding-models/')
fig_path = root_path.joinpath('figure/')

access_token = ''
openai_key = ''
deepseek_api_key = ''

class MyDataParallel(nn.DataParallel):
    def forward(self, *inputs, **kwargs):
        if inputs:
            device = inputs[0].device
        elif kwargs:
            device = next(iter(kwargs.values())).device
        else:
            raise ValueError("No input data found! Ensure batch_question is correctly passed.")

        kwargs = {k: v.to(device) for k, v in kwargs.items()}
        return super().forward(*inputs, **kwargs)

class EarlyStopping:
    def __init__(self, patience=3, mode="max", delta=1e-3):
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.mode = mode  # 'min' for loss, 'max' for accuracy
        self.delta = delta

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
        elif ((self.mode == "min" and current_score > self.best_score - self.delta) or
              (self.mode == "max" and current_score < self.best_score + self.delta)):
            self.counter += 1
            print(f"No improvement. EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = current_score
            self.counter = 0


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def split_and_filter_sentences(text, min_length=10):
    sentences = nltk.sent_tokenize(text)
    filtered_sentences = [sentence for sentence in sentences if len(sentence) >= min_length]
    return filtered_sentences

def split_and_filter_paragraph(text):
    paragraphs = text.split(" \n\n ")
    filtered_paragraphs = [p.strip() for p in paragraphs]
    return filtered_paragraphs

def merge_txt_files_to_string(folder_path):
    all_texts = []

    for root, _, files in os.walk(folder_path):
        for filename in sorted(files): 
            if filename.endswith("Corpus.txt"):
                file_path = os.path.join(root, filename)
                print(f"Open Corpus file: {file_path}")
                with open(file_path, "r", encoding="utf-8") as file:
                    all_texts.append(file.read().strip())  

    return "\n\n".join(all_texts)  

def chunk_by_paragraph(documents):
    paragraph_chunks = []
    for doc in documents:
        text = doc.page_content
        source = doc.metadata

        paragraphs = re.split(r' \n\n ', text.strip())

        for paragraph in paragraphs:
            if paragraph.strip():  
                paragraph_chunks.append(Document(page_content=paragraph, metadata=source))

    return paragraph_chunks

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)    

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extract_log_probs(log_prob):
    """
    Extracts log probabilities from the model's ChoiceLogprobs output.

    Args:
        choice_logprobs (dict): The model's log probability output.

    Returns:
        list: A list of log probabilities for each token.
    """
    return [token_logprob.logprob for token_logprob in log_prob.content]

def calculate_confidence_score(logprobs):
    """
    Calculate the overall confidence score from the model's log probability output.

    Args:
        choice_logprobs (dict): The model's log probability output.

    Returns:
        float: A confidence score between 0 and 1.
    """
    log_probs = extract_log_probs(logprobs)
    
    if not log_probs:
        return 0.0  # Return low confidence if no data
    
    avg_log_prob = np.mean(log_probs)  # Compute the average log probability
    
    # Transform log probability into a confidence score between 0 and 1
    confidence_score = np.exp(avg_log_prob)  
    
    return round(confidence_score, 4)  # Round to 4 decimal places for readability

def parse_json_info(trial_info_str):
    """
    Parses a dictionary-like string containing trial information into a Python dictionary.

    Args:
        trial_info_str (str): A string representation of a dictionary.

    Returns:
        dict: A structured Python dictionary containing the trial information.
    """
    try:
        # Convert string to dictionary using `ast.literal_eval` for safety
        trial_info_dict = ast.literal_eval(trial_info_str)

        # Optionally, validate and normalize the keys/values here
        # Example: Convert 'None' to Python None explicitly
        for key, value in trial_info_dict.items():
            if value == 'None':
                trial_info_dict[key] = None

        return trial_info_dict
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing trial info: {e}")
        return None
    
def get_similarity_from_embedding(index1, index2, normed_total_embeddings):
    embedding_1 = normed_total_embeddings[[index1]]

    embedding_2 = normed_total_embeddings[[index2]]
    cosine_similarities = torch.mm(embedding_1, embedding_2.t()).squeeze(0)

    return cosine_similarities

def calculate_similarity_within_cluster(indices, normed_total_embeddings):
    final_similarity = []
    for i,index1 in enumerate(indices):
        for j,index2 in enumerate(indices[i:]):
            if j > i:
                final_similarity.append(get_similarity_from_embedding(index1,index2,normed_total_embeddings))

    return final_similarity


def compute_cluster_label_weight(similarity, label):
    if label == -1:
        return 1
    elif label == 0:
        return 1
    else:
        return 1

def calculate_mrr(retrieved_paragraphs, evidence_paragraphs):
    for i, paragraph in enumerate(retrieved_paragraphs):
        if paragraph["text"] in evidence_paragraphs:
            return 1 / (i + 1)
    return 0

def print_tensor_memory_usage(tensor_dict, name):
    total_memory = 0
    print(f"\n{name} Memory Usage:")
    for k, v in tensor_dict.items():
        memory = v.element_size() * v.numel() 
        total_memory += memory
        print(f"  {k}: {memory / (1024 ** 2):.2f} MB, device: {v.device}")

    print(f"  Total {name} Memory: {total_memory / (1024 ** 2):.2f} MB\n")
    print("")


def print_gpu_memory(prefix=""):
    print("")
    print(f"{prefix} GPU Memory Usage:")

    num_gpus = torch.cuda.device_count()  
    for gpu_id in range(num_gpus):
        total_memory = torch.cuda.get_device_properties(gpu_id).total_memory
        reserved_memory = torch.cuda.memory_reserved(gpu_id)
        allocated_memory = torch.cuda.memory_allocated(gpu_id)
        free_memory = total_memory - reserved_memory 

        print(f"GPU {gpu_id}:")
        print(f"  Total memory: {total_memory / 1024**2:.2f} MB")
        print(f"  Reserved by PyTorch: {reserved_memory / 1024**2:.2f} MB")
        print(f"  Allocated by PyTorch: {allocated_memory / 1024**2:.2f} MB")
        print(f"  Free memory: {free_memory / 1024**2:.2f} MB")
        print("-" * 40)

def checkpoint_forward(module, *inputs):
    return checkpoint(module, *inputs)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果你在用GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def gpt_json_schema(max_questions=3):
    cluster_format_type = {
        "type": "json_schema",
        "json_schema": {
            "name": "triplet_semantic_similarity_evaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "Reason": {
                        "type": "string",
                        "description": "The explanation for choosing which candidate is semantically more similar to the anchor."
                    },
                    "Token": {
                        "type": "string",
                        "enum": ["|<1>|", "|<2>|"],
                        "description": "The token corresponding to the more semantically similar candidate."
                    }
                },
                "required": ["Reason", "Token"],
                "additionalProperties": False
            }
        }
    }

    cluster_eval_format_type = {
        "type": "json_schema",
        "json_schema": {
            "name": "cluster_semantic_evaluation",
            "schema": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "A brief summary of the main topic or semantic theme of the cluster."
                    },
                    "label": {
                        "type": "string",
                        "enum": ["<STRONG>", "<Medium>", "<WEAK>"],
                        "description": "The level of semantic coherence among the paragraphs in the cluster."
                    }
                },
                "required": ["description", "label"],
                "additionalProperties": False
            }
        }
    }

    generate_question_format_type = {
        "type": "json_schema",
        "json_schema": {
            "name": "paragraph_question_generation",
            "schema": {
                "type": "object",
                "properties": {
                    "questions": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "description": "A clear, concise, and meaningful question based on the paragraph and description."
                                }
                            },
                            "required": ["question"],
                            "additionalProperties": False
                        },
                        "maxItems": max_questions,
                        "description": "List of generated questions, each covering a different but relevant aspect."
                    }
                },
                "required": ["questions"],
                "additionalProperties": False
            }
        }
    }

    return cluster_format_type, cluster_eval_format_type, generate_question_format_type
