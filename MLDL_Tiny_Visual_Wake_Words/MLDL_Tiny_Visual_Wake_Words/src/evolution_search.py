
from exemplar import Exemplar
from random_search import generate_random_network_encode, get_rank_based_on_metrics, get_top_k_models
import random
from metrics.utils_metrics import  isfeasible, compute_metrics
from ea_utils import update_history, prune_population, clean_history
from tqdm import tqdm 

def population_init(N, num_max_blocks,num_min_blocks, max_params, max_flops, inputs, device, fixed_size):

    population = []

    while len(population) < N:
        network_encode = generate_random_network_encode(input_channels_first=3,num_min_blocks=num_min_blocks, num_max_blocks=num_max_blocks, fixed_size=fixed_size)
        exemplar = Exemplar(network_encode, age=0)
        if isfeasible(exemplar,max_params, max_flops, inputs, device):
            population.append(exemplar)
            compute_metrics(exemplar=exemplar, inputs=inputs, device=device)
            print(f"Population size: {len(population)}/{N}")
        else:

            del exemplar
           
    
    return population



def search_evolution(population_size, num_max_blocks,num_min_blocks, max_step, metrics, inputs, device, max_flops, max_params, weight_params_flops=1,fixed_size=False):

    print("Start Evolutionary search ...")
    print("Population initialization ...")
    population = population_init(population_size, num_max_blocks,num_min_blocks,max_params, max_flops, inputs, device, fixed_size)

    #compute_metrics_population(population, inputs, device)
    history = {}
    history = update_history(population, history)

  
    print("Start evolution ...")
    
    for step in tqdm(range(max_step)):
        
        
        sampled = random.sample(population,k=5)
        sampled = get_rank_based_on_metrics(sampled, metrics,weight_params_flops=weight_params_flops)

        parents = get_top_k_models(sampled, k=2)

        # add the children
        for child in mutation(parents, cross=True, age=step+1, max_params=max_params, max_flops=max_flops, inputs=inputs, device=device):
            population.append(child)
            compute_metrics(child, inputs, device)

        #compute_metrics_population(population, inputs, device)
        history = update_history(population, history)

        population = prune_population(population, inputs, device, kill_oldest=True, top_N=population_size, metrics=metrics, 
                                    max_params=max_params, max_flops=max_flops)
            
            
    print("End evolution ...")
    history = clean_history(history, inputs, device, max_params=max_params, max_flops=max_flops)
    final_models = get_rank_based_on_metrics(history.values(), metrics, weight_params_flops=weight_params_flops)
    

    return final_models


def mutation(parents, cross, age, max_params, max_flops, inputs, device):

    """
    Return child models obtained changing the parent models
    """

    while True:
        child_1 = parents[0].mutate()
        if isfeasible(child_1, max_params, max_flops, inputs, device):
            break
        else:
            del child_1
    child_1.set_age(age)

    while True:
        child_2 = parents[1].mutate()
        if isfeasible(child_1, max_params, max_flops, inputs, device):
            break
        else:
            del child_2

    child_2.set_age(age)
    
    if cross:
        while True:
            child_3 = crossover(parents[0], parents[1])
            if isfeasible(child_3, max_params, max_flops, inputs, device):
                break
            else:
                del child_3
                
        child_3.set_age(age)
    
        return child_1, child_2, child_3
    
    return child_1, child_2

def crossover(parent_1, parent_2):

    """
    Return a child model obtained from the parent models
    """

    network_crossover = []
    min_lenghts = min(len(parent_1.network_encode), len(parent_2.network_encode))

    if random.random() < 0.5:
        for i in range(min_lenghts):
            block = random.choice( (parent_1.network_encode[i], parent_2.network_encode[i]))
            network_crossover.append(block)
    else:
        crossover_point = random.randint(0, min_lenghts-1)
        network_crossover = parent_1.network_encode[:crossover_point] + parent_2.network_encode[crossover_point:]
            
    return Exemplar(network_encode=network_crossover)


    
