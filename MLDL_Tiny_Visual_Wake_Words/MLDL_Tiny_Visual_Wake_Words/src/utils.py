
from numpy import random
from metrics.metrics import *
import torch.nn as nn
import matplotlib.pyplot as plt
from search_space import BUILDING_BLOCKS 
from metrics.utils_metrics import compute_metrics_population


channels = [16, 32, 64, 96, 160, 320]
kernel_sizes = [3,5,7]
expansion_factors = [2,4,6]  # for only inverted residual block e ConvNeXt
strides = [1,2]

## Structure of block:
## [ block_type, output_channels, kernel, stride, expansion_factor ]   



def get_rank_based_on_metrics(exemplars, metrics, weight_params_flops=1):
    """
        Based on the indicated metrics, we assign a score to each model. 
        The score is based on the model's position in the total rank (e.g. model -> position 10 -> obtains a score = 10).
        For SynFlow and NASWOT, the model is ranked in ascending order to favor models with high SynFlow and NASWOT values. 
        For FLOPS and #Parameters, the model is ranked in descending order. This way, models with fewer parameters and FLOPS will obtain a better score. 
        For FLOPS and #Parameters, we apply a weight:

            - A weight > 1 gives more importance to FLOPS and parameters than to other metrics.
            - A weight < 1 gives more importance to SynFlow and NASWOT scores than to FLOPS and parameters.
            - weight = 1 gives same importance
    
    """

    scores = {exemplar : 0 for exemplar in exemplars}

    for metric in metrics:

        if metric == "#Parameters" or metric == "FLOPS":
            rank_metric = sorted(exemplars, 
                                 key=lambda x: x.get_cost_info()[0] if metric == "#Parameters" else x.get_cost_info()[1],
                                 reverse=True)

            for i,exemplar in enumerate(rank_metric):
                scores[exemplar] += i*weight_params_flops
        else:

            rank_metric = sorted(exemplars, key=lambda x:  x.get_metric_score(metric))

            for i,exemplar in enumerate(rank_metric):
                scores[exemplar] += i

    final_rank = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)).keys()

    return list(final_rank)



def get_rank_based_fitness(exemplars, metrics, weight_params_flops=1):

    """
    Ranks the model through the fitness function: 
    
    score = synflow/max_synflow + NASWOT/max_NASWOT -w*(params/max_params + flops/max_flops)
    
    """

    tot_scores = {exemplar : 0 for exemplar in exemplars}

    for metric in metrics:

        if metric == "#Parameters":
            scores = [x.get_cost_info()[0] for x in exemplars]
            max_score = max(scores)

            for i,exemplar in enumerate(exemplars):
                score = exemplar.get_cost_info()[0] 
                tot_scores[exemplar] -= weight_params_flops*(score / max_score)
        
        elif metric == "FLOPS":
            scores = [x.get_cost_info()[1] for x in exemplars]
            max_score = max(scores)

            for i,exemplar in enumerate(exemplars):
                score = exemplar.get_cost_info()[1] 
                tot_scores[exemplar] -= weight_params_flops*(score / max_score)
        
        else:

            scores = [x.get_metric_score(metric) for x in exemplars]
            max_score = max(scores)

            for i,exemplar in enumerate(exemplars):
                tot_scores[exemplar] += (exemplar.get_metric_score(metric) / max_score) 

    final_rank = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)).keys()

    return list(final_rank)



def get_top_k_models(networks, k):
    
    if k == 1:
        return networks[0]
    
    return networks[:k]




def generate_random_network_encode(input_channels_first,num_min_blocks=1, num_max_blocks=15, fixed_size=False):

    #input_channels = input_channels_first
    blocks = []

    for _ in range(random.randint(num_min_blocks,num_max_blocks) if fixed_size == False else num_max_blocks):

        block = generate_random_block()
        blocks.append(block)
       
       
    return blocks    
        
def generate_random_block():

    block_type = random.choice(list(BUILDING_BLOCKS.keys()))
    block = generate_random_params(block_type)
    return block 


def generate_random_params(block_type):
     
    kernel_size = 0
    stride = 0
    expansion_factor = 0


    output_channels = random.choice(channels)
    kernel_size = random.choice(kernel_sizes)
    stride = random.choice(strides, p=[0.6, 0.4])
    if block_type == "InvertedResidual" or block_type == "ConvNeXt":
        expansion_factor = random.choice(expansion_factors)
        
    
    return [block_type, output_channels, kernel_size,stride,expansion_factor]
    


def plot_metrics(exemplars, best_exemplars):

    synflow_scores = [exemplar.get_metric_score("synflow") for exemplar in exemplars]
    naswot_scores = [exemplar.get_metric_score("naswot") for exemplar in exemplars]

    best_synflow_scores = [exemplar.get_metric_score("synflow") for exemplar in best_exemplars]
    best_naswot_scores = [exemplar.get_metric_score("naswot") for exemplar in best_exemplars]

    top_1_synflow_score = best_synflow_scores[0]
    top_1_naswot_score = best_naswot_scores[0]

    fig, ax = plt.subplots()
    ax.scatter(naswot_scores, synflow_scores, label="All models")
    ax.scatter(best_naswot_scores, best_synflow_scores, color='orange', label="Top 10 models")
    ax.scatter(top_1_naswot_score, top_1_synflow_score, color="red", label = "Top 1 model")
    ax.set_title("Synflow score vs NASWOT score")
    ax.set_xlabel("NASWOT score")
    ax.set_ylabel("Synflow score")
    plt.legend()
    plt.savefig("Output.jpg")

    return 
