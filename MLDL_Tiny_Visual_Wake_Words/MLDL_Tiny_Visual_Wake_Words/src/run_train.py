from argparse import ArgumentParser
import torch

from dataset import get_data_loader
from train import trainer


parser = ArgumentParser(description="run the model")
parser.add_argument("--model", type=str, default='model.pth')
parser.add_argument("--root_data", type=str, default="COCOdataset/all2017")
parser.add_argument("--ann_train", type=str, default="visualwakewords\instances_train.json")
parser.add_argument("--ann_val", type=str, default="visualwakewords\instances_val.json")
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--resolution_size", type=int, default=224, choices=(96, 128, 160, 192, 224) )
parser.add_argument("--learning_rate", type=float, default=0.01)
parser.add_argument("--momentum", type=float, default=0.9)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--weight_decay", type=float, default=0.000001)
parser.add_argument("--use_subset",type=bool, default=False)
parser.add_argument("--num_batches", type=int, default=100)

args = parser.parse_args()
model = torch.load(args.model)


root_data = args.root_data
path_annotations_train = args.ann_train
path_annotations_val = args.ann_val

# batch size = 64, 96, 128, 256
batch_size = args.batch_size
num_batches = args.num_batches
resolution_size = args.resolution_size
learning_rate = args.learning_rate
momentum = args.momentum
epochs = args.epochs
weigth_decay = args.weight_decay

device = "cuda" if torch.cuda.is_available() else "cpu"
use_subset = args.use_subset


train_dataloader, val_dataloader, test_dataloader = get_data_loader(root_data,
                                                                    path_annotations_train,
                                                                    path_annotations_val,
                                                                    batch_size=batch_size,
                                                                    resolution_size=resolution_size,
                                                                    use_subset=use_subset,
                                                                    num_batches=num_batches)



val_loss, val_accuracy, train_loss, train_accuracy = trainer(train_dataloader,
                                                            val_dataloader,
                                                            test_dataloader,
                                                            learning_rate=learning_rate,
                                                            weight_decay=weigth_decay,
                                                            momentum=momentum,
                                                            epochs=epochs,
                                                            model= model,
                                                            device=device,
                                                            early_stopping=True)
                                     
    

