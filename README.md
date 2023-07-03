# MLDL_Tiny_Visual_Wake_Words


## If you require the data to be downloaded, execute the following command:

```bash
python download_coco_data.py
```
 #To create annotations 
```bash
TRAIN_ANNOTATIONS_FILE="COCOdataset/annotations/instances_train2017.json"
VAL_ANNOTATIONS_FILE="COCOdataset/annotations/instances_val2017.json"
DIR="COCOdataset/annotations/"
!python visualwakewords/scripts/create_coco_train_minival_split.py \
  --train_annotations_file="{TRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="{VAL_ANNOTATIONS_FILE}" \
--output_dir="{DIR}"
```

```bash
MAXITRAIN_ANNOTATIONS_FILE="COCOdataset/annotations/instances_maxitrain.json"
MINIVAL_ANNOTATIONS_FILE="COCOdataset/annotations/instances_minival.json"
VWW_OUTPUT_DIR="visualwakewords"
!python visualwakewords/scripts/create_visualwakewords_annotations.py \
  --train_annotations_file="{MAXITRAIN_ANNOTATIONS_FILE}" \
  --val_annotations_file="{MINIVAL_ANNOTATIONS_FILE}" \
  --output_dir="{VWW_OUTPUT_DIR}" \
  --threshold=0.005 \
  --foreground_class='person'
```
## Run the project
To run the search algorithm, use the following code. The parameters are:
- **algo**: type of algorithm. You can choose between "random_search", "ea_search" (evolutionary algorithm), "our_cnn" (manually design cnn)
- **max_flops**: constraint about number of flops
- **max_params**: constraint about number of parameters
- **metrics**: the metrics implemented are LogSynflow and NASWOT. The options available are "with_cost", means the algorithm will consider the computational cost, "without_cost", the algorithm will consider only the metrics implemented
- **n_random**: number of iterations of random search (needed if you choose "random_search")
- **initial_pop**: initial population size (needed if you choose "ea_search")
- **generation_ea**: number of steps of evolutionary algorithm (needed if you choose "ea_search")
- **max_block**: maximum number of blocks in a model
- **resolution_size**: resolution size of the image, options: 96, 128, 160, 192, 224
- **fixed_size**: if false, the models will not have a fixed number of blocks
- **save**: if True the result model is stored in the file "model.pth"

```bash
python run_search.py \
  --algo ea_search
  --max_flops 200000000
  --max_params 2500000
  --inital_pop 25
  --generation_ea 100
  --save True
```

# To initiate the model training process on the visualwakeword dataset, execute the provided code. The code includes parameters that need to be specified.

- **model**: path of file in which is stored the model ("model.pth" as default)
- **root_data**: path of the dataset folder
- **ann_train**: path of the annotations train file
- **ann_val**: path of the annotations validation file
- **batch_size**: size of the batch for the training phase
- **learning_rate**: learning rate (default 0.01) 
- **momentum**: momentum (default 0.9)
- **epochs:** number of epochs
- **weight_decay**: weight decay (default 1e-4)

```bash
python run_train.py \
  --model "model.pth"
  --root_data "COCOdataset/all2017"
  --ann_train "visualwakewords/instances_train.json"
  --ann_val "visualwakewords/instances_val.json"
  --batch_size 64
  --learning_rate 0.1
  --momentum 0.9
  --epochs 10
  --weight_decay 0.000001
```


