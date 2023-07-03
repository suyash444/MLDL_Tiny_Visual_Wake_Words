
from metrics.metrics import compute_naswot_score, compute_synflow_per_weight, get_params_flops

import torch 

def compute_metrics(exemplar, inputs, device):

    
    if exemplar.metrics == None:

        if not inputs.is_cuda:
            inputs.to(device)

        model = exemplar.get_model()

        if not next(model.parameters()).is_cuda:
            model.to(device)

        exemplar.metrics = {}
        exemplar.metrics["synflow"] = compute_synflow_per_weight(net=model, inputs=inputs, device=device)
        
        exemplar.metrics["naswot"] = compute_naswot_score(net=model, inputs=inputs, device=device)


        inputs.detach()
        del inputs
        model.to("cpu")
        del model
        exemplar.model = None

    return 


def compute_metrics_population(population, inputs, device):

    for exemplar in population:
        compute_metrics(exemplar=exemplar, inputs=inputs, device=device)

    return 
    
def isfeasible(exemplar, max_params: int, max_flops: int, inputs, device): 

    if exemplar.params == None or exemplar.flops == None:
        exemplar.params, exemplar.flops = get_params_flops(exemplar.get_model(), inputs, device)

    if exemplar.flops <= max_flops and exemplar.params <= max_params:
        return True
    else: 
        del exemplar
        return False