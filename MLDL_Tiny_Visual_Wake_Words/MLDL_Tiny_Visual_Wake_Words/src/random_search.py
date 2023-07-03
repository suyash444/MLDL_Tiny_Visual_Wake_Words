from search_space import *
import random
import math 
from exemplar import Exemplar
from utils import generate_random_network_encode, get_rank_based_on_metrics, get_top_k_models, get_rank_based_fitness
from metrics.utils_metrics import  isfeasible, compute_metrics
from tqdm import tqdm

def search_random(num_iterations, num_max_blocks,num_min_blocks, max_params, max_flops, input_channels_first, \
                   k, metrics,weight_params_flops, inputs, device, fixed_size=False, fitness_func=False):

    print("Start random search ...")
    population = []

   
    for i in tqdm(range(num_iterations)):
            
        network_encoded = generate_random_network_encode(input_channels_first=input_channels_first,num_min_blocks=num_min_blocks, num_max_blocks=num_max_blocks, fixed_size=fixed_size)
            
        exemplar = Exemplar(network_encoded)

        if isfeasible(exemplar, max_params=max_params, max_flops=max_flops, inputs=inputs, device=device):
            compute_metrics(exemplar, inputs, device)
            exemplar.model = None
            population.append(exemplar)
        else:
            del exemplar
             
    
    print("Finish random search.")
    print(f"Remaining {len(population)} that satisfy constraints")
    
    if fitness_func:
        population_rank = get_rank_based_fitness(population, metrics, weight_params_flops=weight_params_flops)
    else:
        population_rank = get_rank_based_on_metrics(population, metrics, weight_params_flops=1)
   
    return population_rank

    

        

