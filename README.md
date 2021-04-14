# Galaxy Classification Project

## For Testing

### Data

For first experiments I created a smaller dataset with 2500 images you can find in galaxy_dev/
The data is split into train, val and test sets vis prefixes (this is more natural for Pegasus)

For HPO we use:
train set with 1750 images
val set with 250 images

### Scripts

You need these scripts for the experiments

* model_selection.py - has definition of the pretrained model and early stopping modules
* data_loader.py  - custom dataset loader is defined here
* vgg16_hpo.py - experiment driver


RUN 

```python
python vgg16_hpo.py --trials 3 --epochs 5
```

The script creates a number of artifacts: checkpoints, plots and txt with best results.


```
/checkpoints/vgg16_galaxy/    (study object checkpoint and early stopping weights for all of the trials)
/exp_results_details/timestamp/         (here we get loss function plot and txt with best HPO from all trials)  
```



This model overfitts. You can get tranining acuracy that is higher than validation accuracy very fast. There has to be data augmentation added.
The final version will be training on bigger more diverse dataset (where augmentation techniques like horizontal flip, jitter and so on are added).

```
    
Galaxy Classification

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        batch size for training
  --cuda CUDA           use gpu support
  --seed SEED           select seed number for reproducibility
  --root_path ROOT_PATH
                        path to dataset
  --save SAVE           path to checkpoint save directory
  --epochs EPOCHS       number of training epochs
  --trials TRIALS       number of HPO trials                      (default: 2)
  --ex_rate EX_RATE     info exchange rate in HPO                 (default: 2)

```

## General Instructions 
### Step 1: Download Dataset

https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge

### Step 2: Run Dataset Generating Script

```python
python create_dataset.py
```
### Step 3: Split Dataset into Train, Test and Validation Set

```python
python split_data.py
```
### Step 4: Tune Model with Optuna

```python
python vgg16_hpo.py
```


### Decaf and Pegasus
![img](Galaxy-Decaf.png)

### Confusion Matrix
![img](confusion_matrix_unnorm.png)
