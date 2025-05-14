import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packages import *
from utils import mean_pooling
from utils import calculate_confidence_score
from AgentComponents.AgentWorker import DeepSeekWorker

def evaluate_cluster(corpus_dict):
    system_prompt = '''
You will be provided with a ditonary of text paragraphs, where the keys are indices and values are texts. 
Please read them carefully and group them into semantic clusters based on their topics or meanings.
Assign each paragraph to a cluster and provide a short description for each cluster. The clusters should reflect distinct topics or themes present in the paragraphs.
Also return a label in <STRONG>, <Medium>, <WEAK> to indicating the relationship within each cluster.

Please provide your response in the following format:

[{"indices": [list of paragraph numbers], "description": short description, "label": "<STRONG>"}, {"indices": [list of paragraph numbers], "description": short description, "label": "<Medium>"}]

Note:
1. Only return the list of dictionaries as required,do not return anything else.
2. The number of clusters are flexible, but make sure each of the cluster contains paragraphs with very similar semantic meanings.
3. Each cluster must have at least 2 paragraphs.
4. Double check to make sure that the clutser index is matching the input accurately.
'''
        
    # Create input
    input_text = f"Paragraphs to be evaluted:\n {str(corpus_dict)}"
    
    # Generate response
    generation_flag, attempt_count = False, 0
    data = None
    while not generation_flag and attempt_count < 5:
        try:
            agent = DeepSeekWorker()
            answer, log_prob = agent.generate(question=input_text, system_prompt=system_prompt, return_log_prob=True)
            confidence_score = calculate_confidence_score(log_prob)
            data = ast.literal_eval(answer)
            if data is not None:
                generation_flag = True
        except Exception as e:
            print(f"Error evaluating cluster: {e}")
            attempt_count += 1
    
    # Return data if successful
    if generation_flag:
        return data, confidence_score

    return data, confidence_score