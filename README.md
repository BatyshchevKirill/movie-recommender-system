# Practical Machine Learning and Deep Learning - Assignment 2 - Movie Recommender System

## Student Info

**Full Name**: Kirill Batyshchev  
**Email Address**: k.batyshchev@innopolis.university  
**Group Number**: B21-DS-02

## Installation Steps

The following instructions are to be executed using **Python v3.11**.

#### 1. Clone the Repository

```console
git clone https://github.com/BatyshchevKirill/movie-recommender-system.git
```

#### 2. Change the directory to the project root
```console
movie-recommender-system
```
#### 3. Install Required Python Libraries

```console
pip install -r requirements.txt
```

## Data Processing
### Dataset processing
Download the dataset using the following script

```bash
python download_dataset.py
```
After that you need to preprocess dataset using the following script
```bash
python preprocess.py
```
If you want to preprocess some other data, you can state the following parameters:<br>
-d: folder with data<br>
-s: folder where you want the outputs to be saved<br>
-tr: training file name<br>
-t: test file name

## Model Training

To train a model, execute the training script.

```bash
python train.py
```
You may specify the following parameters (optionally, they are now set to default values):<br>
-d: base directory path<br>
-a: alpha parameter<br>
-b: batch size<br>
-id: input dimension of autoencoder<br>
-hd: hidden dimension of autoencoder<br>
-ed: encoded dimension of autoencoder<br>
-e: epoch number<br>
-n: noise weight<br>
-l1: lasso regularization weight<br>
-l2: ridge regularization weight<br>
-c: model checkpoint will be saved here<br>
-n: number of clusters in kmeans<br><br>
By default models are saved to models/. The resulting mapping table and users' clusters will be saved
to data/interim/ with names rating.csv, encoded_ae_[some numbers].csv and clustered_ae_[some numbers].csv

## Prediction
The trained model example is on the github, or you can train your own one. 
To make a prediction run the following script
```bash
python predict.py [eval|rec] USER_ID -u path/to/user/cluster.csv -r path/to/model/result.csv -m path/to/movies.csv -f film_id -k reccomendation_number -mn lower_bound_foir_rating -mx upper_bound_for_rating
```
However, almost all arguments have default value, so you have to specify only
the user's id for recommendation mode (rec) or user's id and film id for rating
prediction mode (eval). Example:
```bash
python predict.py rec 1
```

## Result Testing
You can test the result of prediction or recommendation of a model using the following script
```bash
python benchmark/evaluate.py [rmse|pr] -u path/to/user_clusters.csv -t path/to/test/data.csv -r path/to/model/results.csv
```
The default values are already set to everything, so you can simply test it by running, for example:
```bash
python benchmark/evaluate.py pr
```
