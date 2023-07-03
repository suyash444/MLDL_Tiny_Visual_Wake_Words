from torch.utils.data import RandomSampler
from torch.utils.data.sampler import SubsetRandomSampler
import torch
import random
import pyvww
from torchvision import transforms as T
from timm.data import create_transform

def get_data_loader(root_data, path_annotations_train,path_annotations_val,batch_size, test_batch_size=256,resolution_size=224,use_subset=False, num_batches=100):
  
  """
  Returns data loaders of train set, validation set and test set.
  """
  
  
  transform_training = create_transform(input_size=resolution_size,is_training=True, auto_augment='rand-m6-mstd0.5')
  transform_testing = create_transform(input_size=resolution_size,is_training=False, auto_augment='rand-m6-mstd0.5')
  
  full_training_data = pyvww.pytorch.VisualWakeWordsClassification(
    root=root_data,
    transform=transform_training,
    annFile=path_annotations_train
  )

  test_data = pyvww.pytorch.VisualWakeWordsClassification(
    root=root_data,
    transform=transform_testing,
    annFile=path_annotations_val
  )

  


  if use_subset:
    num_samples = num_batches*batch_size
    subset_training_data = []
    for i in range(num_samples):
      image,label = full_training_data[random.randint(0, len(full_training_data))]
      subset_training_data.append( (image,label))
      
  else:
    num_samples = len(full_training_data)  
  
  # Indices for the split dataset: Split full_train_dataset into train and val dataset
  training_samples = int(num_samples*0.8+1)
  validation_samples = num_samples - training_samples     

  training_data, validation_data = torch.utils.data.random_split(
      subset_training_data if use_subset == True else full_training_data,
      [training_samples, validation_samples]
  ) 

  # Initialize dataloader   
  train_loader = torch.utils.data.DataLoader(training_data, batch_size, shuffle=True, num_workers=4)
  val_loader = torch.utils.data.DataLoader(validation_data, batch_size, shuffle=False, num_workers=4)
  test_loader = torch.utils.data.DataLoader(test_data, test_batch_size, shuffle=False, num_workers=4)

  return train_loader, val_loader, test_loader
