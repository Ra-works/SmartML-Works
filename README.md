## MLOps framework

The below are the comprehensive steps must needed for 
a model training on any dataset.

Use the model_params file under config section for any
hyper param definitions

### Data 
Data preparation
Data preprocessing
Data cleansing
Data wrangling

Data Encoding
Data scaling
Data normalization

#### Visualize Data

Feature Engineering
Clustering
Dimensionality Reduction

#### Visualize Data

k-fold define
CrossValidation Preparation

### Model

TrainModel
  |- DecisionTree
  |- Boosting
  |- SVM (SUPPORT VECTOR MACHINES)
  |- KNN (K-NEAREST NEIGHBOURS)
  |- NN (NEURAL NETWORKS)

Score Logging
  |- Accuracy
  |- Precision
  |- Recall
  |- F1Score

#### Visualize Score

Score visualization 
  |- Train score plot
  |- Validation score plot

Score analysis
  |- Bias Analysis
  |- Variance Analysis
  |- Overfitting Analysis
  |- Underfitting Analysis

#### Visualize time
Time analysis
  |- Training time
  |- Validation time
  |- Epoch Time

Save the model checkpoints
Version the models

Retrain the models

** Repeat

Select the Model
Version the model
Save the checkpoint

########

Deploy to Production

