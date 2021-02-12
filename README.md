# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

The dataset that is used for the implementation of this project is UCI Bank Marketing Dataset. This dataset contains 20 features and 10000 rows. The problem is a classification problem in which we have to predict whether the client will subscribe to a term deposit or not.

The problem is solved in two different ways, i.e., using MS Azure Hyperdrive and MS Azure AutoML. The maximum accuracy is achieved by the best-performing automl model, i.e., VotingEnsemble model which is 91.62%. 

## Scikit-learn Pipeline

The Scikit-Learn pipeline construction involves the following main steps:
  1. First of all, the data is imported from this [url](https://automlsamplenotebookdata.blob.core.windows.net/automl-sample-notebook-data/bankmarketing_train.csv) using Azure *TabularDatasetFactory* predefined method .
  2. The imported data is then converted into a pandas dataframe and preprocessed i.e null records are removed, categorical features are one-hot encoded.
  3. The clean data is then divided into train and test dataset, with 0.2 is chosen as the test data size ratio.
  4. The classification algorithm that is used for the Scikit-Learn Pipeline is *Logistic Regression*.
  5. The hyperparameters of Logistic Regression that are tuned using Hyperdrive are *Regularization Strength* and *Maximum number of Iterations*. 
  6. The *RandomParameterSampling* is used to select the suitable values for *C* and *max_iter* hyperparameters from the following search spaces.   
      '--C': choice(0.001, 0.01, 0.1, 1, 10, 100)  
      '--max_iter': choice(range(25, 500, 25))  
  7. *BanditPolicy* is used as an early termination policy with a slack factor of 0.1.
    
**What are the benefits of the parameter sampler you chose?**

Random Sampling is fast compared to Grid Search Sampling. It randomly picks hyperparameters from the given search space instead of applying a brute force search over all the available parameters as in Grid Search. In this way, it is computationally efficient and thus cost-effective.

**What are the benefits of the early stopping policy you chose?**

Early termination policy is specified to automatically terminates poorly performing runs. Bandit Policy, used in this project, is based on slack factor/amount and evaluation interval. It terminates run where the primary metric is not within the specified slack factor/amount compared to the best performing run. In this way, it improves computational efficiency.

## AutoML

In AutoML run, 35 models are trained out of which the best performing model is Voting Ensemble model, formed by a combination of various other models including XGBoostClassifier, LightGBM, LogisticRegression and RandomForest. This model gave an accuracy of 91.62%. 

The hyperparameters of one of the xgboostclassifier are as follows:   
{'base_score': 0.5,  
 'booster': 'gbtree',  
 'colsample_bylevel': 1,  
 'colsample_bynode': 1,  
 'colsample_bytree': 0.5,  
 'eta': 0.3,  
 'gamma': 0,  
 'learning_rate': 0.1,  
 'max_delta_step': 0,  
 'max_depth': 6,  
 'max_leaves': 0,  
 'min_child_weight': 1,  
 'missing': nan,  
 'n_estimators': 100,  
 'n_jobs': 1,  
 'nthread': None,  
 'objective': 'reg:logistic',  
 'random_state': 0,  
 'reg_alpha': 1.7708333333333335,  
 'reg_lambda': 2.5,  
 'scale_pos_weight': 1,  
 'seed': None,  
 'silent': None,  
 'subsample': 0.7,  
 'tree_method': 'auto',  
 'verbose': -10,  
 'verbosity': 0}  
 
The ensemble weights and algorithms for the trained model are: 

*ensemble_weights* : \[0.07692307692307693, 0.07692307692307693, 0.3076923076923077, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.07692307692307693, 0.15384615384615385\] 

*ensembled_algorithms* : \['XGBoostClassifier', 'LightGBM', 'XGBoostClassifier', 'XGBoostClassifier', 'XGBoostClassifier', 'RandomForest', 'LightGBM', 'LogisticRegression', 'LightGBM'\]  


## Pipeline comparison

The maximum accuracy achieved by Hyperdrive logistic regression model is 91.22% whereas the best performing automl model i.e. VotingEnsemble model achieved an overall accuracy of 91.62%. 

The main difference between the Hyperdrive and AutoML approaches used in this project is that in the Hyperdrive approach, we need to select the Machine Learning algorithm, which is Logistic Regression in this case, before tuning its hyperparameters with the Hyperdrive package. Whereas in the AutoML approach, all the steps from selecting a suitable machine learning model for the problem to the hyperparameters tuning and model training is achieved by AutoML.

Moreover, we have used cross validation in AutoML run but we didn't use cross validation in hyperdrive run. 

## Future work

In this project, Accuracy is used as the primary metric of performance. However, accuracy is not a very good metric of performance for classification problems when there is class imbalance in the dataset. In future we can use class imbalance handling techniques such as SMOTE, alongwith better performance metrics such as precision, recall to determine whether the model is not overfitting and performing well.

## Proof of cluster clean up
![](Cluster%20Clean%20Up%20Proof.png)
