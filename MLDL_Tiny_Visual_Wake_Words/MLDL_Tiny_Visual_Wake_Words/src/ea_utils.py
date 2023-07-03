
from exemplar import Exemplar
from metrics.utils_metrics import compute_metrics_population, isfeasible
from math import inf
from utils import get_rank_based_on_metrics, get_top_k_models

def create_history(exemplars):
    history = {exemplar.network_encode : exemplar for exemplar in exemplars}
    return history

def update_history(exemplars, history):
    
    # add new exemplars
    history.update( {encode_model(exemplar.network_encode) : exemplar for exemplar in exemplars if encode_model(exemplar.network_encode) not in history})
    return history

def clean_history(history, inputs, device, max_params, max_flops):
    history = { network_encode : exemplar for network_encode, exemplar in history.items()
               if isfeasible(exemplar, max_params, max_flops, inputs, device)}
    return history


def prune_population(population, inputs, device,kill_oldest=True, top_N = 25, metrics=['synflow', 'naswot'], max_params=25*(10**5), max_flops=200*(10**6) ):

    
    population_feasible = [exemplar for exemplar in population if isfeasible(exemplar, max_params, max_flops, inputs, device)]

    min_age = inf 
    idx_oldest = None

    if kill_oldest:
        for i,exemplar in enumerate(population_feasible):
            if exemplar.age < min_age:
                min_age = exemplar.age
                idx_oldest = i
        
        population_feasible.pop(idx_oldest)

    population_feasible_scored = get_rank_based_on_metrics(population_feasible, metrics=metrics)
    top_k_exemplars = get_top_k_models(population_feasible_scored, top_N)
    return top_k_exemplars

def analyze_history(history, metrics):

    history_models = get_rank_based_on_metrics(list(history.values()), metrics)
    top_models = get_top_k_models(history_models, k=3)
    return top_models



def encode_model(block_list):
    
    block_str = ""
    for block in block_list:
        block_str += f"({block[0]}, {block[1]}, {block[2]}, {block[3]}, {block[4]});"
    
    return block_str