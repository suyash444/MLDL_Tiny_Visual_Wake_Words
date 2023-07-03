from argparse import ArgumentParser
import torch
from random_search import search_random
from evolution_search import search_evolution

from models.mobileNetV3 import MobileNetV3
from metrics.metrics import get_params_flops, compute_synflow_per_weight, compute_naswot_score
from decimal import Decimal
import random
from utils import plot_metrics, get_top_k_models

def main():
    

    random.seed(0)

    parser = ArgumentParser()
    
    parser.add_argument("--algo", type=str, default='random_search', choices=("random_search", "ea_search","MobileNetV3"))
    parser.add_argument('--max_flops', type=float, default=200*(10**6))
    parser.add_argument('--max_params', type=float, default=25*(10**5))
    parser.add_argument('--metrics', type=str, default="with_cost", choices=("without_cost", "with_cost"))
    parser.add_argument('--n_random', type=int, default=20)
    parser.add_argument('--initial_pop', type=int, default=5)
    parser.add_argument('--generation_ea', type=int, default=30)
    parser.add_argument('--max_blocks', type=int, default=7)
    parser.add_argument('--resolution_size', type=int, default=224, choices=(96, 128, 160, 192, 224))
    parser.add_argument('--fixed_size',type=bool, default=False)
    parser.add_argument("--save", type=bool, default=False)


    device = "cuda" if torch.cuda.is_available() else "cpu"


    args = parser.parse_args()

    max_flops = args.max_flops
    max_params = args.max_params
    max_blocks = args.max_blocks
    fixed_size = args.fixed_size
    resolution_size = args.resolution_size

    mini_batch_size = 32
    
    if device == "cpu":
         inputs = torch.rand(mini_batch_size,3,resolution_size, resolution_size).to(device)
    else:
        inputs = torch.rand(mini_batch_size,3,resolution_size, resolution_size).type(torch.float16).to(device)

    if args.metrics == "without_cost":
        metrics = ['synflow', 'naswot']
    else:
        metrics = ['synflow','naswot','FLOPS','#Parameters']

    if args.algo == "ea_search":
        population_size = args.initial_pop
        num_generations = args.generation_ea
        rank_models = search_evolution(population_size=population_size,
                                       num_max_blocks=max_blocks,
                                       num_min_blocks=3,
                                       max_step=num_generations,
                                       metrics=metrics,
                                       inputs=inputs,
                                       device=device,
                                       max_flops=max_flops,
                                       max_params=max_params,
                                       weight_params_flops=1,
                                       fixed_size=fixed_size)
        
        best_exemplars = get_top_k_models(rank_models,10 if len(rank_models) > 10 else len(rank_models))
        top_1_exemplar = best_exemplars[0]
        top_1_model = top_1_exemplar.get_model()
        plot_metrics(rank_models, best_exemplars)
    
    elif args.algo == "random_search":
        num_models = args.n_random
        rank_models = search_random(num_iterations=num_models,
                                    num_max_blocks=max_blocks,
                                    num_min_blocks=3,
                                    max_params=max_params,
                                    max_flops=max_flops,
                                    input_channels_first=3,
                                    k=3,
                                    metrics=metrics,
                                    weight_params_flops=1,
                                    inputs=inputs,
                                    device=device,
                                    fixed_size=fixed_size)
        
        best_exemplars = get_top_k_models(rank_models,10 if len(rank_models) > 10 else len(rank_models))
        top_1_exemplar = best_exemplars[0]
        top_1_model = top_1_exemplar.get_model()
        plot_metrics(rank_models, best_exemplars)

    elif args.algo == "MobileNetV3":
        top_1_model = MobileNetV3(mode='small', classes_num=2, input_size=32, width_multiplier=0.35)
        params, flops = get_params_flops(top_1_model, inputs, device)
        inputs, top_1_model = inputs.to(device), top_1_model.to(device)
        naswot = compute_naswot_score(top_1_model, inputs=inputs, device=device)
        synflow = compute_synflow_per_weight(top_1_model, inputs, device)
        print(f"MobileNetV3 --- \n params = {params} flops = {flops} \n")
        print(f"Synflow score: {synflow}\nNASWOT score: {naswot}")


    if args.save:
      torch.save(top_1_model, f'Model_{args.algo}.pth')   
        
    #print best model
    if args.algo == "random_search" or args.algo == "ea_search":
        print("Best exemplar obtained ---")
        print(f"Model: {top_1_model}")
        print(f"#Parameters: {top_1_exemplar.get_cost_info()[0]}  FLOPS: {top_1_exemplar.get_cost_info()[1]}")
        print(f"Synflow score: {top_1_exemplar.get_metric_score('synflow')}")
        print(f"NASWOT score: {top_1_exemplar.get_metric_score('naswot')}")

    
    return 



if __name__ == '__main__':
    main()


