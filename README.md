# Optimizing an ML Pipeline in Azure

## Overview

This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary

The dataset that is used for the implementation of this project is UCI Bank Marketing Dataset. This dataset contains 20 features and 10000 rows. The problem is a classification problem in which we have to predict whether the client will subscribe to a term deposit or not.

The problem is solved in two different ways, i.e., using MS Azure Hyperdrive and MS Azure AutoML. The maximum accuracy is achieved by the best-performing automl model, i.e., VotingEnsemble model which is 91.65%. 

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

In AutoML run, 38 models are trained out of which the best performing model is Voting Ensemble model which gave an accuracy of 91.65%.
A subset of the hyperparameters obtained from the best performing model is:  
{  
        objective='reg:logistic',<br />
        random_state=0,<br />
        reg_alpha=1.0416666666666667,<br />
        reg_lambda=1.5625,<br />
        scale_pos_weight=1,<br />
        seed=None, <br />
        silent=None,<br />
        subsample=0.8,<br />
        tree_method='hist',<br />
        verbose=-10,<br />
        verbosity=0<br />
}<br />

The weights of the trained model are: 

weights=\[0.25, 0.08333333333333333, 0.25, 0.16666666666666666, 0.08333333333333333,0.08333333333333333, 0.08333333333333333\], 

## Pipeline comparison

The maximum accuracy achieved by Hyperdrive logistic regression model is 91.22% whereas the best performing automl model i.e. VotingEnsemble model achieved an overall accuracy of 91.65%. 

The main difference between the Hyperdrive and AutoML approaches used in this project is that in the Hyperdrive approach, we need to select the Machine Learning algorithm, which is Logistic Regression in this case, before tuning its hyperparameters with the Hyperdrive package. Whereas in the AutoML approach, all the steps from selecting a suitable machine learning model for the problem to the hyperparameters tuning and model training is achieved by AutoML.

Moreover, we have used cross validation in AutoML run but we didn't use cross validation in hyperdrive run. 

## Future work

In this project, Accuracy is used as the primary metric of performance. However, accuracy is not a very good metric of performance for classification problems when there is class imbalance in the dataset. In future we can use better performance metrics such as precision, recall.  

