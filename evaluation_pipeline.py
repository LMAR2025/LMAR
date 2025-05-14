from packages import *
from utils import library_path, output_path, vectorstore_path, model_output_path, table_path
from utils import split_and_filter_sentences, merge_txt_files_to_string, split_and_filter_paragraph, chunk_by_paragraph
from utils import load_json, save_json
from utils import calculate_confidence_score, get_similarity_from_embedding, calculate_similarity_within_cluster, compute_cluster_label_weight, calculate_mrr
from utils import (access_token, openai_key, deepseek_api_key)

from AgentComponents.AsyncApiRequests import LLMQueue, OpenAIGPTClient, DeepSeekClient, execute_llm_tasks, run_async_llm_tasks
from AgentComponents.AgentWorker import DeepSeekWorker
from AgentComponents.EmbedderModule import EmbedderModule
from AgentComponents.ClusterEvalAgent import evaluate_cluster

from sklearn.model_selection import KFold, train_test_split

from langchain_community.document_loaders import TextLoader
from AgentComponents.embedding_finetune import MyDataset, mean_pooling, split_batch, tfidf_overlap_similarity, get_random_content, create_dataset
from AgentComponents.VectorStore import VectorStore

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for dataset processing.")
    parser.add_argument("--dataset_name", type=str, default="techQA", help="Name of the dataset")
    parser.add_argument("--model_type", type=str, default="Base", help="Choose to run on trained or baseline model")
    parser.add_argument("--model_name", type=str, default="Linq-AI-Research/Linq-Embed-Mistral", help="Choose baseline model")
    parser.add_argument("--exp_name", type = str, default='no_cluster_update', help = 'The name of experiments')
    
    args = parser.parse_args()
    return args

def run_eval_pipeline(dataset_name, model_type = 'Base', model_save_path=None, model_name = None, training_model = None, training_epoch = None, exp_name = '', llm_type="",max_len = 1500, lr= "no_lr", save=True,
                      top_k_list = [5, 10, 20, 30]):

    # Mannully Setting Part
    dataset_path = library_path.joinpath(dataset_name)
    # model_save_path = library_path.joinpath(f'{dataset_name}/embedding_model/{model_name}_{exp_name}/')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create a local embedding model template
    model_name = model_name
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if model_type == "Trained":
        #  model = AutoModel.from_pretrained(model_save_path,device_map = 'auto')
        assert model_save_path is not None, "model_save_path cannot be None."
        model = AutoModel.from_pretrained(model_save_path, trust_remote_code=True).to(device)
        print("Local Model Loaded")
    if model_type == "Base":
        model = AutoModel.from_pretrained(model_name, quantization_config = BitsAndBytesConfig(load_in_8bit=True), device_map = 'auto')
        # model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        print("Using Base Model")
    if model_type == "Training":
        print("Using Training Model")
        model = training_model
    
    print_gpu_memory("Model Loaded:")


    if dataset_path.joinpath(f"Corpus_all.txt").exists() and dataset_path.joinpath(f"Queries_all.json").exists():
        loader = TextLoader(dataset_path.joinpath(f"Corpus_all.txt"), encoding="utf-8")
        with open(dataset_path.joinpath(f"Queries_all.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        loader = TextLoader(dataset_path.joinpath(f"Corpus.txt"), encoding="utf-8")
        with open(dataset_path.joinpath(f"Queries.json"), "r", encoding="utf-8") as f:
            data = json.load(f)

    documents = loader.load()
    texts = chunk_by_paragraph(documents)

    question_list = [item['Question'] for item in data]
    evidence_list = [item['Evidence'] for item in data]

    token_lengths = [len(tokenizer.tokenize(evidence)) for evidence in evidence_list]
    print(f"Average length of evidence token: {np.mean(token_lengths)}")
    print(f"Standard deviation of token length: {np.std(token_lengths)}")

    model.eval()
    with torch.no_grad():
        VS = VectorStore(model=model, tokenizer=tokenizer, train=True, device=device, max_length = max_len) # Set to training model without saving the results locally
        VS.load_new_text(texts)
        VS.embed_library(batch_size=10)

    # top_k_list = [5, 10, 20, 30]
    # top_k_list = [2,3,4,5]
    results = []

    with torch.no_grad():
        VS.embedding = VS.embedding.to(device)
        for top_k in top_k_list:
            TFScore = []
            Corrects = []
            mrr_scores = []

            for i in tqdm(range(len(question_list)), desc=f"Evaluating Top-{top_k}"):
                # TFScore
                retrieved_paragraphs_lst = VS.retrieve(query=question_list[i], top_k=top_k)[1]
                retrieved_paragraphs = " \n ".join(j["text"] for j in retrieved_paragraphs_lst)

                TFScore.append(tfidf_overlap_similarity(retrieved_paragraphs, evidence_list[i]))

                # Accuracy (hit or not)
                correct = int(any(j["text"] in evidence_list[i] for j in retrieved_paragraphs_lst))
                Corrects.append(correct)

                # MRR
                mrr_scores.append(calculate_mrr(retrieved_paragraphs_lst, evidence_list[i]))

            # 计算指标
            mrr_pct = np.percentile(mrr_scores, q=[2.5, 50, 97.5])
            TFScore_pct = np.percentile(TFScore, q=[2.5, 50, 97.5])
            accuracy_pct = np.mean(Corrects)

            results.append({
                "Model": model_name,
                "Epoch": training_epoch,
                "top_k": top_k,
                "Accuracy": accuracy_pct,
                "MRR_2.5": mrr_pct[0],
                "MRR_50": mrr_pct[1],
                "MRR_97.5": mrr_pct[2],
                "MRR_mean": np.mean(mrr_scores),
                "TFScore_2.5": TFScore_pct[0],
                "TFScore_50": TFScore_pct[1],
                "TFScore_97.5": TFScore_pct[2],
                "TFScore_mean": np.mean(TFScore),
            })

        eval_df = pd.DataFrame(results)

        if len(top_k_list) > 1:

            embedder = EmbedderModule(base_model=model)

            embedding_batch_size = 10
            # Embedding questions

            for i in tqdm(range(0,len(question_list), embedding_batch_size), desc="Embedding Questions"):
                embeddings = embedder.embed(question_list[i:i+embedding_batch_size])
                if i == 0:
                    question_total_embeddings = embeddings.cpu()
                else:
                    question_total_embeddings = torch.concat([question_total_embeddings, embeddings.cpu()],dim = 0)

            question_normed_total_embeddings = question_total_embeddings.clone()
            for i in tqdm(range(question_total_embeddings.shape[0]), desc="Normalization Questions"):
                question_normed_total_embeddings[i] = question_total_embeddings[i]/torch.norm(question_total_embeddings[i], p = 2, dim = 0, keepdim=True)

            # Embedding evidence
            for i in tqdm(range(0,len(evidence_list), embedding_batch_size), desc="Embedding Evidences"):
                embeddings = embedder.embed(evidence_list[i:i+embedding_batch_size])
                if i == 0:
                    evidence_total_embeddings = embeddings.cpu()
                else:
                    evidence_total_embeddings = torch.concat([evidence_total_embeddings, embeddings.cpu()],dim = 0)

            evidence_normed_total_embeddings = evidence_total_embeddings.clone()
            for i in tqdm(range(evidence_total_embeddings.shape[0]), desc="Normalization Evidences"):
                evidence_normed_total_embeddings[i] = evidence_total_embeddings[i]/torch.norm(evidence_total_embeddings[i], p = 2, dim = 0, keepdim=True)

            cosine_similarities = torch.sum(question_normed_total_embeddings*evidence_normed_total_embeddings,dim = 1)
            average_similarity = torch.mean(cosine_similarities)
            eval_df['Average Similarity'] = average_similarity.item()

    # return eval_df
    library_dataset_path = table_path.joinpath(dataset_name)
    library_dataset_path.mkdir(parents=True, exist_ok=True)

    # return eval_df

    if training_epoch:
        save_path = library_dataset_path.joinpath(
            f'{dataset_name}_{model_type}_{exp_name}_{llm_type}_eval_table_epoch{training_epoch}_{lr}.xlsx')
    else:
        save_path = library_dataset_path.joinpath(f'{dataset_name}_{model_type}_{exp_name}_{llm_type}_eval_table_{lr}.xlsx')

    if save:
        eval_df.to_excel(save_path, index=False)
        print(f"Evaluation table successfully saved to: {save_path}")

    return eval_df


if __name__ == '__main__':
    args = parse_arguments()
    # model_name = "Linq-AI-Research/Linq-Embed-Mistral"
    # model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens" # √
    # model_name = "BAAI/bge-m3" # √
    # eval_df = run_eval_pipeline(dataset_name="techQA", model_type="Base", model_name=model_name, training_model=None, training_epoch=None, exp_name='base', max_len=1500)
    eval_df = run_eval_pipeline(dataset_name=args.dataset_name, model_type=args.model_type, model_name=args.model_name,training_model=None, training_epoch=None, exp_name='base', max_len=512)
    print(eval_df[["top_k", "Accuracy", "MRR_mean", 'Average Similarity']])
    print("Finished")
