
from packages import *
from utils import library_path, output_path, vectorstore_path, model_output_path 
from utils import split_and_filter_sentences, merge_txt_files_to_string, split_and_filter_paragraph, chunk_by_paragraph, set_seed
from utils import load_json, save_json
from utils import calculate_mrr, calculate_confidence_score, get_similarity_from_embedding, calculate_similarity_within_cluster, compute_cluster_label_weight, print_tensor_memory_usage, print_gpu_memory
from utils import (access_token, openai_key, deepseek_api_key)
from utils import MyDataParallel, EarlyStopping
from utils import gpt_json_schema

from AgentComponents.AsyncApiRequests import LLMQueue, OpenAIGPTClient, DeepSeekClient, execute_llm_tasks, run_async_llm_tasks
from AgentComponents.AgentWorker import DeepSeekWorker
from AgentComponents.EmbedderModule import EmbedderModule
from AgentComponents.ClusterEvalAgent import evaluate_cluster

from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch import autocast
# from torch.amp import GradScaler
from torch.cuda.amp import GradScaler

from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Subset

from langchain_community.document_loaders import TextLoader
from AgentComponents.embedding_finetune import MyDataset, mean_pooling, split_batch, tfidf_overlap_similarity, get_random_content, create_dataset, scaled_similarity_loss
from AgentComponents.VectorStore import VectorStore

from evaluation_pipeline import run_eval_pipeline

def parse_arguments():
    parser = argparse.ArgumentParser(description="Argument parser for dataset processing.")
    parser.add_argument("--dataset_name", type=str, default="techQA", help="Name of the dataset")
    parser.add_argument("--testing_mode", type=bool, default=False, help="Enable or disable testing mode")
    parser.add_argument("--model_name", type=str, default="BAAI/bge-m3", help="Choose baseline model")
    parser.add_argument("--exp_name", type = str, default='main', help = 'The name of experiments')
    parser.add_argument("--llm_type", type=str, default = 'DeepSeek', help = 'The name of LLM used'  )
    args = parser.parse_args()
    return args

def run_pipeline(dataset_name, testing_mode = False, model_name="BAAI/bge-m3", exp_name='', llm_type = ''):
    # Define Dataset Name and other Hyperparameters
    # Argparser Part
    dataset_name = dataset_name
    testing_mode = testing_mode
    # set_seed(42)

    print(f"dataset_name: {dataset_name}")
    print(f"testing_mode: {testing_mode}")
    print(f"model_name: {model_name}")
    print(f"exp_name: {exp_name}")
    print(f"llm_type: {llm_type}")

    # Mannully Setting Part
    dataset_path = library_path.joinpath(dataset_name)
    model_save_path = library_path.joinpath(f'{dataset_name}/embedding_model/{model_name}_{exp_name}/')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Create a local embedding model template
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Training Hyper Params
    cluster_epochs = 50
    qa_epochs = 50
    max_len = 512
    shuffle = True
    batch_size = 32
    cluster_batch_size = 16
    warmup_steps = 100
    cluster_lr = 1e-6
    qa_lr = 1e-6
    test_topk = 15
    max_question_num=3
    llm_type = llm_type


    # Loading data
    ## Corpus
    combined_paragraph_str = merge_txt_files_to_string(dataset_path)
    corpus_list = split_and_filter_paragraph(combined_paragraph_str)
    ## Queries
    queries = load_json(dataset_path.joinpath("Queries.json"))
    ## All Text
    loader = TextLoader(library_path.joinpath(dataset_name, f"Corpus.txt"), encoding="utf-8")
    documents = loader.load()
    texts = chunk_by_paragraph(documents)

    cluster_format_type, cluster_eval_format_type, generate_question_format_type = gpt_json_schema(
        max_questions=max_question_num)

    print(f"Length of corpus_list: {len(corpus_list)}")
    # sys.exit()

    ## Start Below:
    if 'embedding_model' in os.listdir(dataset_path):
        embedding_model = AutoModel.from_pretrained(model_save_path, trust_remote_code=True).to(device)
        assert "No embedding saved now"
    else:
        embedding_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

        embedding_model.gradient_checkpointing_enable()
        # embedding_model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)

    assert embedding_model.device.type == 'cuda', "Embedding Model are not deployed on GPU."

    ###########################################################################################################################
    # Sample Cluster
    ###########################################################################################################################

    embedder = EmbedderModule(base_model=embedding_model, max_len=max_len)

    # Embedding all texts
    embedding_batch_size = 10

    for i in tqdm(range(0,len(corpus_list), embedding_batch_size), desc="Embedding Progress"):
        embeddings = embedder.embed(corpus_list[i:i+embedding_batch_size])
        if i == 0:
            total_embeddings = embeddings.cpu()
        else:
            total_embeddings = torch.concat([total_embeddings, embeddings.cpu()],dim = 0)

    normed_total_embeddings = total_embeddings.clone()
    for i in tqdm(range(total_embeddings.shape[0]), desc="Normalization Progress"):
        normed_total_embeddings[i] = total_embeddings[i]/torch.norm(total_embeddings[i], p = 2, dim = 0, keepdim=True)

    sample_size = 1300
    top_k_sample = 10
    cluster_list = []
    available_idx = range(normed_total_embeddings.shape[0])

    start_time = time.time()

    for i in tqdm(range(sample_size)):
        # Sample Anchor Index
        idx = np.random.choice(available_idx,1)

        # Get Anchor Embedding
        query_embedding = normed_total_embeddings[idx]

        # Calculate Cosine Similarity
        available_embeddings = normed_total_embeddings
        cosine_similarities = torch.mm(query_embedding, available_embeddings.t()).squeeze(0)

        # Sample from top-K most similar embeddings
        top_k_similarities, top_k_indices = torch.topk(cosine_similarities, k=top_k_sample+1, largest=True)

        top_k_similarities = top_k_similarities[1:]
        top_k_indices = top_k_indices[1:]

        samples = np.random.choice(top_k_indices,2,replace=False)
        cluster_list.append({'indices':list(idx) + list(samples)})

    print("Finish Sample Cluster!!")
    ##########################################################################################################################
    # Generate Triplet
    ##########################################################################################################################
    system_prompt_cluster = '''
            You will be provided with a triplet of texts. There is an anchor text and two candidate text paragraphs. Your task is to determine which one is more similar to the anchor text semantically.
            Each Candidate is labeled with special token |<1>| or |<2>|. Please first given reason for your decision and return corresponding special label at the end.

            Note:
            1. Please first given reason for your decision and return a corresponding special label in a dictionary format.
            2. First analyze on overall topic consistency, and then consider the context, entities, or event.
            JSON format Example Out:
            {
            "Reason": "The reason for choosing the first candidate text |<1>| is that it described the potential solutions to the same problem as the anchor text",
            "Token": "|<1>|"
            }
        '''

    user_prompt_cluster_lst = [f"Anchor Text:\n {corpus_list[cluster['indices'][0]]} \n Candidate Text |<1>|: {corpus_list[cluster['indices'][1]]} \n Candidate Text |<2>|: {corpus_list[cluster['indices'][2]]}" for cluster in cluster_list]

    llm_queue = execute_llm_tasks(system_prompt_cluster, user_prompt_cluster_lst, return_log_prob=False,
                                          response_format=cluster_format_type, batch_size=1, num_workers=1,
                                          llm_type=llm_type, start = 0, end = len(user_prompt_cluster_lst))

    usage_dict = {"total_token": llm_queue.total_token, "input_token": llm_queue.input_token,
                  "output_token": llm_queue.output_token}
    print(f"Generate Triplet Usage: {usage_dict}")

    cluster_sample_df = pd.DataFrame()
    for task_id, value in llm_queue.task_results.items():
        try:
            index = int(task_id.split("_")[1])
            answer = value["response"]
            positive_index = int(answer['Token'][2])
            negative_index = [x for x in [1,2] if x != positive_index][0]
            tmp_df = pd.DataFrame({'Anchor': corpus_list[cluster_list[index]['indices'][0]], 'Positive': corpus_list[cluster_list[index]['indices'][positive_index]], 'Negative':corpus_list[cluster_list[index]['indices'][negative_index]]},index = [0])
            cluster_sample_df = pd.concat([cluster_sample_df, tmp_df],axis=0, ignore_index=True)
        except Exception as e:
            print(f"Error Detail : {e}")

    cluster_sample_df.to_csv(dataset_path.joinpath(f'triplet_samples_{llm_type}.csv'), index = False)

    print("Finish Generate Triplet!!")
    ###########################################################################################################################
    #### Training Clustering
    ###########################################################################################################################
    cluster_sample_df = pd.read_csv(dataset_path.joinpath(f'triplet_samples_{llm_type}.csv'))
    print_gpu_memory("Before Training")
    # Before Training GPU Memory: Allocated=10668.42 MB, Reserved=12172.00 MB

    total_batch_num = np.ceil(cluster_sample_df.shape[0]/cluster_batch_size)
    total_steps =  total_batch_num * cluster_epochs
    cluster_model = embedding_model
    optimizer = AdamW(cluster_model.parameters(), lr=cluster_lr, fused=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    best_loss = float('inf')
    early_stopping = EarlyStopping(patience=3, mode='min',delta=1e-2)

    scaler = GradScaler()
    trip_loss = nn.TripletMarginLoss(margin=0.2, p=2)

    cluster_save_path = model_output_path.joinpath(dataset_name, model_name.split("/")[-1], "cluster_model")
    cluster_save_path.mkdir(parents=True, exist_ok=True)

    for epoch in range(cluster_epochs):
        cluster_model.train()
        total_loss = 0

        progress_bar = tqdm(enumerate(range(0,cluster_sample_df.shape[0],cluster_batch_size)), total=total_batch_num,desc=f'Training Epoch {epoch+1}')
        for _, i in progress_bar:
            batch_df = cluster_sample_df.loc[i:i+cluster_batch_size-1,:].copy().reset_index(drop = True)
            # print(batch_df.shape[0])
            anchor = batch_df['Anchor'].to_list()
            positive = batch_df['Positive'].to_list()
            negative = batch_df['Negative'].to_list()

            with autocast(device_type='cuda'):
                anchor_input = tokenizer(anchor, padding='max_length', truncation=True, return_tensors='pt',
                                         max_length=max_len).to(device)
                positive_input = tokenizer(positive, padding='max_length', truncation=True, return_tensors='pt',
                                           max_length=max_len).to(device)
                negative_input = tokenizer(negative, padding='max_length', truncation=True, return_tensors='pt',
                                           max_length=max_len).to(device)

                # print_gpu_memory("Loaded Input")
                anchor_output = cluster_model(**anchor_input)
                positive_output = cluster_model(**positive_input)
                negative_output = cluster_model(**negative_input)

                # print_gpu_memory("Loaded Embedding")
                anchor_embeddings = mean_pooling(anchor_output, anchor_input['attention_mask'])
                positive_embeddings = mean_pooling(positive_output, positive_input['attention_mask'])
                negative_embeddings = mean_pooling(negative_output, negative_input['attention_mask'])

                cluster_loss = trip_loss(anchor_embeddings, positive_embeddings, negative_embeddings)

            optimizer.zero_grad(set_to_none=True)
            # print_gpu_memory("Before qaloss backward")
            scaler.scale(cluster_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += cluster_loss.detach().item()
            progress_bar.set_postfix({'Loss': cluster_loss.detach().item()})

            del batch_df
            del anchor_input, anchor_output, anchor_embeddings
            del positive_input, positive_output, positive_embeddings
            del negative_input, negative_output, negative_embeddings
            del cluster_loss
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        avg_loss = total_loss / total_batch_num
        print(f'Average Loss for Epoch {epoch + 1}: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            cluster_model.save_pretrained(cluster_save_path)  # 你需要定义 cluster_save_path
            print(f"Saved new best model at {cluster_save_path} with Loss = {avg_loss:.4f}")

        early_stopping(avg_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered. Best Loss = {best_loss:.4f}")
            break


    print("Finish Training Clustering!!")

    eval_df = run_eval_pipeline(dataset_name=dataset_name, model_type="Trained", model_save_path=cluster_save_path, model_name=model_name, training_epoch=epoch,
                                     llm_type=llm_type, exp_name="cluster", max_len=max_len, lr=cluster_lr, save=True, top_k_list = [2,3,4,5])

    print("Cluster Evaluation")
    print(eval_df[["top_k", "Accuracy", "MRR_mean", 'Average Similarity']])

    cluster_model = AutoModel.from_pretrained(cluster_save_path, trust_remote_code=True).to(device)
    ##########################################################################################################################
    # Re-Embed the corpus & Generating Clustering Sample
    ##########################################################################################################################
    embedder = EmbedderModule(base_model=cluster_model, max_len = max_len)

    # Embedding all texts
    embedding_batch_size = 10

    for i in tqdm(range(0,len(corpus_list), embedding_batch_size), desc="Re-Embedding Progress"):
        embeddings = embedder.embed(corpus_list[i:i+embedding_batch_size])
        if i == 0:
            total_embeddings = embeddings.cpu()
        else:
            total_embeddings = torch.concat([total_embeddings, embeddings.cpu()],dim = 0)

    normed_total_embeddings = total_embeddings.clone()
    for i in tqdm(range(total_embeddings.shape[0]), desc="Normalization Progress"):
        normed_total_embeddings[i] = total_embeddings[i]/torch.norm(total_embeddings[i], p = 2, dim = 0, keepdim=True)

    # Resample Cluster
    max_cluster_size = 3
    threshold = 0.15
    available_idx = list(range(0,normed_total_embeddings.shape[0]))
    cluster_list = []
    choosed_idx = []

    start_time = time.time()

    while len(available_idx) > 0:
        idx = np.random.choice(available_idx,1)
        query_embedding = normed_total_embeddings[idx]

        available_embeddings = normed_total_embeddings[available_idx]
        cosine_similarities = torch.mm(query_embedding, available_embeddings.t()).squeeze(0)

        top_k_similarities, top_k_relative_indices = torch.topk(cosine_similarities, k=min(max_cluster_size, len(available_idx)), largest=True)
        raw_last_similarity = top_k_similarities[-1].item()

        filter = torch.where(1-top_k_similarities < threshold)
        top_k_similarities_filtered = top_k_similarities[filter]
        top_k_relative_indices_filtered = top_k_relative_indices[filter]

        # Map back to original indices
        top_k_indices = [available_idx[int(i)] for i in top_k_relative_indices_filtered.tolist()]
        top_k_similarities = top_k_similarities_filtered.tolist()


        cluster = {'indices': top_k_indices, 'similarity': top_k_similarities, 'raw_last_similarity': raw_last_similarity}
        cluster_list.append(cluster)
        choosed_idx += top_k_indices
        available_idx = [x for x in available_idx if x not in top_k_indices]

    print('Clustering Finished in ', time.time() - start_time)
    print(f"Length of cluster list: {len(cluster_list)}")
    cluster_sizes = [len(cluster['indices']) for cluster in cluster_list]
    average_cluster_size = sum(cluster_sizes) / len(cluster_sizes)
    print(f"Average cluster size: {average_cluster_size}")
    raw_last_similarities = [cluster['raw_last_similarity'] for cluster in cluster_list]
    mean_similarity = np.mean(raw_last_similarities)
    std_similarity = np.std(raw_last_similarities)
    print(f"Average raw_last_similarity: {mean_similarity:.4f}; Standard deviation of raw_last_similarity: {std_similarity:.4f}")

    ##########################################################################################################################
    # Generate Description
    ##########################################################################################################################
    start_time = time.time()

    system_prompt_cluster_eval  = '''
        You will be provided with a list of text paragraphs that belong to the same pre-defined cluster.
        Please carefully read these paragraphs and generate:
        1. A brief description summarizing the main topic or semantic theme of the cluster.
        2. A label indicating the semantic coherence among the paragraphs in the cluster. Choose one of the following labels: <STRONG>, <Medium>, or <WEAK>.

        Return your response in the following JSON format:

        JSON Example output:
        {
            "description": "Brief summary of the topic",
            "label": "<STRONG>"
        }

        You must start your response **with a left curly brace {**, and return a valid JSON object.
        End your response **exactly** at the closing `}`. Do not include anything else.

        Notes:
        1. Only return the result in the specified JSON format. Do not include any additional explanation or text.
        2. Make sure the description accurately reflects the shared topic of the cluster, and the label appropriately represents the level of semantic similarity among the paragraphs.
    '''

    user_prompt_cluster_eval_lst = [
        f"Paragraphs to be evaluated:\n {str({i: corpus_list[i] for i in cluster['indices']})}" for cluster in
        cluster_list]

    if testing_mode:
        results = execute_llm_tasks(system_prompt_cluster_eval, user_prompt_cluster_eval_lst[:100], return_log_prob=False,
                                            response_format=cluster_eval_format_type, batch_size=1, num_workers=1,
                                            llm_type=llm_type, start = 0, end = len(user_prompt_cluster_eval_lst))
    else:
        results = execute_llm_tasks(system_prompt_cluster_eval, user_prompt_cluster_eval_lst, return_log_prob=False,
                                            response_format=cluster_eval_format_type, batch_size=1, num_workers=1,
                                            llm_type=llm_type, start = 0, end = len(user_prompt_cluster_eval_lst))

    usage_dict = {"total_token":results.total_token, "input_token":results.input_token, "output_token":results.output_token}
    print(f"Generate Description Usage: {usage_dict}")

    all_cluster_evaluation = []
    for task_id, value in results.task_results.items():
        try:
            json_output = value["response"]
            index = int(task_id.split("_")[1])
            output = {
                "indices": cluster_list[index]["indices"],
                "description": json_output["description"],
                "label": json_output["label"]
            }
            all_cluster_evaluation.append(output)
        except TypeError as e:
            print(f"TypeError for task_id {task_id}: {e}")

    print('Clustering Evaluation finished in ', time.time() - start_time)

    save_json(all_cluster_evaluation,dataset_path.joinpath(f'{dataset_name}_{llm_type}_cluster_LLM_eval.json'))

    # ##########################################################################################################################
    # # Generate S-Q Pair
    # ##########################################################################################################################
    print('Generate S-Q pairs.')
    cluster_res = load_json(dataset_path.joinpath(f"{dataset_name}_{llm_type}_cluster_LLM_eval.json"))
    cluster_list = [' \n\n '.join(corpus_list[j] for j in item['indices'] if j <= len(corpus_list)) for i, item in enumerate(cluster_res)]

    system_prompt_SQ = """
    You are an AI that generates structured JSON output.
    Your task is to generate questions according to given paragraphs and descriptions.
    Your response **must** follow this JSON format exactly:

    {{
        "questions": [
            {{"question": "..."}},
            {{"question": "..."}}
        ]
    }}


    You must start your response **with a left curly brace {{**, and return a valid JSON object.
    End your response **exactly** at the closing `}}`. Do not include anything else.

    ### Rules:
    - Generate up to {max_question_num} questions.
    - Each question should be **clear, concise, and meaningful**.
    - The questions must be strictly based on the content of the provided description and paragraph, and should cover different, but relevant aspects of the description.
    - Each question should be answerable using the given paragraph as evidence.
    - **Do NOT include explanations, only return JSON output.**
    """.format(max_question_num=max_question_num)

    user_prompt_SQ_lst = [f" Paragraphs:\n" + '\n'.join(corpus_list[j] for j in item['indices'] if j <= len(corpus_list)) +
                          f"\n\nDescription:\n{item['description']}\n\nGenerate up to {max_question_num} questions."
                          for i, item in enumerate(cluster_res)]

    llm_queue = execute_llm_tasks(system_prompt_SQ, user_prompt_SQ_lst, return_log_prob=False,
                           response_format=generate_question_format_type, batch_size=1, num_workers=1,
                           llm_type=llm_type, start = 0, end = len(user_prompt_SQ_lst))

    usage_dict = {"total_token": llm_queue.total_token, "input_token": llm_queue.input_token,
                  "output_token": llm_queue.output_token}
    print(f"Generate S-Q Pair Usage: {usage_dict}")

    # Format Outputs
    all_outputs = {}
    for task_id, value in llm_queue.task_results.items():
        try:
            json_questions = value["response"]
            index = int(task_id.split("_")[1])
            all_outputs[task_id] = {
                "Description": cluster_res[index]["description"],
                "Evidence": cluster_list[index],
                "Questions": json_questions["questions"]}
        except TypeError as e:
            print(f"TypeError for task_id {task_id}: {e}")

    # Save AQ pair data
    AQ_path = dataset_path.joinpath(f"{dataset_name}_AQ_{exp_name}_{llm_type}.json")
    with open(AQ_path, "w", encoding="utf-8") as f:
        json.dump(all_outputs, f, ensure_ascii=False, indent=4)
    print(f"Finish writing AQ json to {AQ_path}")

    ###########################################################################################################################
    # # Format Training Data
    ###########################################################################################################################

    question_json = load_json(dataset_path.joinpath(f"{dataset_name}_AQ_{exp_name}_{llm_type}.json"))
    # question_json = load_json(dataset_path.joinpath(f"{dataset_name}_AQ_no_cluster.json"))
    print(dataset_path.joinpath(f"{dataset_name}_AQ_{exp_name}_{llm_type}.json"))
    print(len(question_json))

    question_df = pd.concat([
        pd.DataFrame([
            {
                'Evidence': item['Evidence'],
                'Question': q['question']
            }
            for q in item['Questions']
        ])
        for item in question_json.values()
    ], ignore_index=True)

    train_df, test_df = train_test_split(question_df, test_size=0.5, shuffle=True)

    # all_dataset = create_dataset(question_df, tokenizer, max_len=max_len)
    train_dataset = create_dataset(train_df, tokenizer, max_len=max_len, VS=None, use_vs_negatives=False)
    print(f"Length of all dataset: {len(train_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    total_batch_num = len(train_dataloader)

    test_question_list = test_df["Question"].tolist()
    test_evidence_list = test_df["Evidence"].tolist()


    ###########################################################################################################################
    # #### Training
    ###########################################################################################################################
    print_gpu_memory("Before Training")
    # Before Training GPU Memory: Allocated=10668.42 MB, Reserved=12172.00 MB
    total_steps = len(train_dataloader) * qa_epochs
    model = cluster_model
    optimizer = AdamW(model.parameters(), lr=qa_lr, fused=True)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    early_stopping = EarlyStopping(patience=3, mode='max')
    best_mrr = 0.0
    save_path = model_output_path.joinpath(dataset_name, model_name.split("/")[-1], "_".join([llm_type, exp_name]))
    save_path.mkdir(parents=True, exist_ok=True)

    scaler = GradScaler()
    cos_loss = nn.CosineEmbeddingLoss()

    for epoch in range(qa_epochs):
        model.train()
        total_loss = 0

        batch_counter = 0
        # accumulated_loss = torch.tensor(0.0, device=device, requires_grad=True)

        progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader),desc=f'Training Epoch {epoch+1}')

        for i, (batch_data, batch_labels) in progress_bar:
            # QA Loss
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)

            batch_question, batch_context = split_batch(batch_data)

            # print_gpu_memory(f"Before Training Batch {i}")
            with autocast(device_type='cuda'):
                # print_gpu_memory("Before batch")
                question_output = model(**batch_question)
                # print_gpu_memory("After question forward")
                context_output = model(**batch_context)
                # print_gpu_memory("After context forward")

                question_embeddings = mean_pooling(question_output, batch_question['attention_mask'])
                context_embeddings = mean_pooling(context_output, batch_context['attention_mask'])

                qa_loss = cos_loss(question_embeddings, context_embeddings, batch_labels)

            optimizer.zero_grad(set_to_none=True)
            # print_gpu_memory("Before qaloss backward")
            scaler.scale(qa_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            total_loss += qa_loss.detach().item()
            progress_bar.set_postfix({'Loss': qa_loss.detach().item()})

            del batch_data, batch_labels, question_output, context_output
            del question_embeddings, context_embeddings, qa_loss
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

            batch_counter += 1

        model.eval()

        eval_df = run_eval_pipeline(dataset_name=dataset_name, model_type="Trained", model_save_path=save_path,
                                    model_name=model_name, training_epoch=epoch, llm_type=llm_type,
                                    exp_name=exp_name, max_len=max_len, lr=qa_lr, save=False, top_k_list=[test_topk])

        mrr_score = eval_df["MRR_mean"][0]

        print(f"MRR_mean for Epoch {epoch+1}: {mrr_score}")

        if mrr_score > best_mrr:
            best_mrr = mrr_score
            model.save_pretrained(save_path)
            print(f"Saved new best model at {save_path} with MRR = {mrr_score:.4f}")

        early_stopping(mrr_score)
        if early_stopping.early_stop:
            print(f"Saved new best model at {save_path} with MRR = {mrr_score:.4f}")
            break

    print('Start Evaluation')
    eval_df = run_eval_pipeline(dataset_name=dataset_name, model_type="Trained", model_save_path=save_path, model_name=model_name, training_epoch=epoch,
                                     llm_type=llm_type, exp_name=exp_name, max_len=max_len, lr=qa_lr, top_k_list = [2,3,4,5])

    print("All Evaluation")
    print(eval_df[["top_k", "Accuracy", "MRR_mean", 'Average Similarity']])

if __name__ == '__main__':
    args = parse_arguments()
    run_pipeline(dataset_name = args.dataset_name, testing_mode = args.testing_mode, model_name=args.model_name, exp_name=args.exp_name, llm_type=args.llm_type)
    # model_name = "sentence-transformers/distilbert-base-nli-stsb-mean-tokens"
    # run_pipeline(dataset_name = "pubmedQA", testing_mode = False, model_name=model_name, exp_name="with_cluster", llm_type="Llama")

    print("Finished!")