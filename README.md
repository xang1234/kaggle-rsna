# RSNA Intracranial Hemorrhage Detection

This model is based on [Appian's kernel](https://github.com/appian42/kaggle-rsna-intracranial-hemorrhage)


## Environment

Refer to environment.yaml


## Preparation

Please put `./input` directory in the root level and unzip the file downloaded from kaggle there. All other directories such as `./cache`, `./data`, `./model` will be created if needed when `./bin/preprocess.sh` is run.


## Steps to produce predictions

### 1. Preprocessing

Please make sure you run the script from parent directory of `./bin`. Run the following once :

~~~
$ sh ./bin/preprocess.sh
~~~

preprocess.sh does the following at once.

- dicom_to_dataframe.py(src/preprocess/dicom_to_dataframe.py) reads dicom files and save its metadata into the dataframe. 
- create_dataset.py(src/preprocess/create_dataset.py) creates a dataset for training.
- make_folds.py(src/preprocess/make_folds.py) makes folds for cross validation. 


### 2. Training

There are 5 different model binaries and config files as per the table below. To run replace `x` with the desired model number. The desired fold has to be manually changed to the correct value in the `train00x.sh` file
~~~
$ sh ./bin/train00x.sh
~~~

| Model | Folds |  Backbone | Window Policy | Image size | 
----|----|----|----|----
| 001 | 0,1,2,3,4 | se\_resnext50\_32x4d | 2 |512x512 | 
| 003 | 0,2,4 | efficientnet-b3 |2 |512x512 |
| 003_2 | 3 | efficientnet-b3 | 3 |512x512 |
| 004 | 0,1,2,3, | inceptionV4 |2 |512x512 |
| 005 | 2 | efficientnet-b5 | 2 |512x512 |

- running efficietnnet-b3 on fold 1 will produce an error during 1st epoch validation
- only model003_2 uses Window Policy=3 which adds CLAHE preprocessing from scikit-image

### 3. Predicting

~~~
$ sh ./bin/predict00x.sh
~~~

predict00x.sh does the predictions and makes a submission file for scoring on Kaggle. Each model has as separate .sh file but the fold number has to be manually updated each time. 

### 4. Ensembling 

- `ensemble1.py` (src/ensemble/ensemble1.py) ensembles all the models above to produce a submission file. The path to the file has to be updated accordingly
- `ensemble2.py` (src/ensemble/ensemble2.py) ensembles all the models with public lb<=0.71 to produce a submission file. The path to the file has to be updated accordingly


### 5. Retraining and ensembling 

Once the public test data is released, retraining can be done if time permits by repeating step 2,3 and 4 (the data path has to be updated accordingly. Such files have the `_retrain` suffix. The config files will be reused

| Model | Folds |  Backbone | Window Policy | Image size | 
----|----|----|----|----
| 001_retrain | 0,1,2,3,4 | se\_resnext50\_32x4d | 2 |512x512 | 
| 003_retrain | 0,2,4 | efficientnet-b3 |2 |512x512 |
| 003_2_retrain | 3 | efficientnet-b3 | 3 |512x512 |
| 004_retrain | 0,1,2,3, | inceptionV4 |2 |512x512 |
| 005_retrain | 2 | efficientnet-b5 | 2 |512x512 |

Retraining allows us to generate 4 additional submissions

- `ensemble3.py` (src/ensemble/ensemble1.py) ensembles all the `retrained` models above to produce a submission file. The path to the file has to be updated accordingly
- `ensemble4.py` (src/ensemble/ensemble2.py) ensembles all the `retrained` models with public lb<=0.71 to produce a submission file as per ensemble2.py. The path to the file has to be updated accordingly (selection to be done based on Stage 1 public lb score only - Stage 2 public lb is likely to be too small anyway)
- `ensemble5.py` (src/ensemble/ensemble1.py) ensembles all the `initial and retrained` models above to produce a submission file. The path to the file has to be updated accordingly
- `ensemble6.py` (src/ensemble/ensemble2.py) ensembles all the `initial and retrained` models with public lb<=0.71 to produce a submission file as per ensemble2.py. The path to the file has to be updated accordingly (selection to be done based on Stage 1 public lb score only - Stage 2 public lb is likely to be too small anyway)

If there is enough retraining time I will probably select ensemble 5 and 6 , otherwise I'll stick with 1 and 2 for the final submission. 
