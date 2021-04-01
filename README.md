# Galaxy Classification Project

## For Testing

### Data

For first dev I created a smaller dataset with 1250 images you can find in galaxy_dev/

### Scripts

You need these scripts for the experiments

* model_selection.py - has definition of the pretrained model and early stopping modules
* data_loader.py  - custom dataset loader is defined here
* vgg16_hpo.py - experiment driver


RUN 

```python
python vgg16_hpo.py
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




