import numpy as np
from exemplar import Exemplar
import torch 

# prob = [2/6, 3/6, 1/6]          # change a block, change params of a block, add a block
# result = np.random.multinomial(1, prob)
# idx = np.argmax(result)

# print(result)
# print(idx)

network_encode = [ ["InvertedResidual", 32], 
                   [ "DWConv", 16]]

parent = Exemplar(network_encode)
print("parent")
print(parent.get_model())

print("\n\n\n child \n\n\n")
child = parent.mutate()

model = child.get_model()
print(model)

input = torch.rand(1,3,224,224)
label = model(input)
print(label)
