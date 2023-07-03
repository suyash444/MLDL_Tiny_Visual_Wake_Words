
from copy import deepcopy
import torch
from search_space import NetworkDecoded
from utils import generate_random_block, generate_random_params
import random
import numpy as np

class Exemplar:

    def __init__(self, network_encode, age=0):         # add other attributes for evolution
        super().__init__()

        self.network_encode = network_encode
        self.metrics = None
        self.params = None
        self.flops = None
        self.model = None 
        self.age = 0

    def get_metric_score(self, metric):
        return self.metrics[metric]

    def get_model(self):
        
        if self.model == None:
            self.model = NetworkDecoded(self.network_encode, num_classes=2)
        
        return self.model
    
    def get_cost_info(self):
        return self.params, self.flops
    
    def mutate(self, random=True):
        
        """
        Mutation of the model. Options: 
        - change a block
        - cut a block
        - add a block
        """
        
        probs = [1/3, 1/3, 1/3]
        chosen = np.random.multinomial(n=1, pvals=probs)
        chosen = np.argmax(chosen)
        
        # choose random idx of block
        if random:
            idx_block = np.random.randint(0, high=len(self.network_encode))
            
        new_network_encode = deepcopy(self.network_encode)

        # Change a block
        if chosen == 0:       
            new_network_encode[idx_block] = generate_random_block()
        # cut a block
        elif chosen == 1:
            for i in range(idx_block, len(self.network_encode)-1):
                new_network_encode[i] = new_network_encode[i+1]
        # add a block
        else:
            new_network_encode.append( generate_random_block() )

        return Exemplar(new_network_encode, age=0)
    
    def set_age(self,age):
        self.age = age 
    
    
        
        


    