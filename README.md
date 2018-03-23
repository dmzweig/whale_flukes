# Humpback whale recognition 

*Authors: David Zweig and Marc Negre,
The repository structure and an important proportion of the code 
comes from Olivier Moindrot and Guillaume Genthial in their
[tutorials](https://cs230-stanford.github.io) (Hand Signs Recognition with Tf)*


## Prepare the whale image dataset
The original images correspond to 20,060 images of whale tails,
9,850 are labelled, 15,610 are non-labelled and dedicated to the test set.

images are named following where the label is in `[0, 5]`.
The training set contains 1,080 images and the test set contains 120 images.

Run the script `build_dataset_whale.py` which will resize the images to size `(64, 64)`:

```bash
python build_dataset.py --data_dir (your data dir) --output_dir data/64x64_NUMBER_LABELS_WHALES_NONW
```

Run the script `changeNameFiles.py` which will rename the files following `{label}_IMG_{id}.jpg` 
where the label is in `[0, 4250; 5000]`. 5000 corresponds to the 'new_whale' label
that now needs to be moved out of the training folder.

## Augment the data

data_aug.py generates additional images to supplement the initial dataset by performing a series of random rotations, zooms, and shifts to the input image. It should be run after the dataset has been renamed and the initial dataset built, using only the training images as an input. After creating the augmented images, they should be combined with the source images to make the complete training dataset. It takes as inputs a data_dir, the source of the images to be augmented, and an int_dir,the directory to which the images should be saved before combination with the original set:

```bash
python data_aug.py --data_dir data/train_whales --int_dir data/train_whales_augments


## Quickstart (~10 min) [Points 2-5, (c) Olivier Moindrot & Guillaume Genthial]

1. __Build the dataset of size 64x64__: follow the instructions above "Prepare the whale image dataset"

2. __Your first experiment__ We created a `base_model` directory for you under the `experiments` directory. It countains a file `params.json` which sets the parameters for the experiment. It looks like
```json
{
    "learning_rate": 1e-3,
    "batch_size": 32,
    "num_epochs": 10,
    ...
}
```
For every new experiment, you will need to create a new directory under `experiments` with a similar `params.json` file.

3. __Train__ your experiment. Simply run
```
python train.py --data_dir data/64x64_NUMBER_LABELS_WHALES_NONW --model_dir experiments/base_model
```
It will instantiate a model and train it on the training set following the parameters specified in `params.json`. It will also evaluate some metrics on the development set.

4. __Your first hyperparameters search__ We created a new directory `learning_rate` in `experiments` for you. Now, run
```
python search_hyperparams.py --data_dir data/64x64_NUMBER_LABELS_WHALES_NONW --parent_dir experiments/learning_rate
```
It will train and evaluate a model with different values of learning rate defined in `search_hyperparams.py` and create a new directory for each experiment under `experiments/learning_rate/`.

5. __Display the results__ of the hyperparameters search in a nice format
```
python synthesize_results.py --parent_dir experiments/learning_rate
```

7. __Prediction on the test set__ Because the Kaggle challenge doesn't provide labels on the test set,
we have to run predictions and output the results in a .csv file with the following format:

"Image,Id
image.jpg,prediction_1 prediction_2 prediction_3 prediction_4 prediction_5"

The predictions will be written in the kaggle_submissions folder.

```
python predict.py --model_dir experiments/(your experiment) --data_dir data/64x64_NUMBER_LABELS_WHALES_NONW
```

8. __Perform triplet loss__ (unable to reach completion at the end of the project, unfortunately)
Several files have been created to attempt at building the data structure for triplet. 

`model/input_fn_triplet.py`
`model/model_fn_triplet.py`
`train_triplet.py`
`trip_pipeline_1.py`

## Presentation of the label_files 
This folder contains all the necessary .txt and .csv files that link the image.jpg names to their labels. The labels have either 
their string name (for instance: 'w_7554f44') or their number labels (range [0:4249; 5000], 5000 corresponds to the 'new_whale' 
that we seperated out from the rest of the training dataset).



