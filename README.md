# home-care duration prediction by machine learning methods

Data are provided by Buurtzorg company, which is one of the biggest home-care organization in Netherlands. Data are based on 2017 year. 

In this case study, machine learning methods are applied to predict home-care duration for next week. Multiple classifiers are tried but they all failed. But Gradient-boosted trees(GBTs) and Deep neural networks(DNN) for regression perform quite well and they all reach nearly 0.99 coefficient of determination(R2). GBTs got better results on mean absolute error(MAE) 0.2130 and Root Mean Square Error(RMSE) 9.0385 , however, DNN only got MAE 7.0359 and RMSE 13.1125. But RMSE variance of DNN(19.9206) is much lower than GBTs(36.3063). So GBTs will predict more accurate care-duration, but it is sensitive to outliers while DNN is much more stable.

## File structure

### Plot: EDA. 

* Scatter plot about all predictors and the response. 
* Distribution plot of the response. 
* Correlation matrix plot about the preprocessed data. 

### Data: 
This directory contains all data files, which are in Rds format.

### Pre-processing-R: 
This directory contains files for preprocessing data.

### Classification: 
Files about predicting categorized care duration.
Classfiication performance is worse than regression. Trained classifiers are logistic regression, decision tree, random forest.  Classifiers were trained on Spark (Databricks platform).


### Regression: 
Files about predicting continuous care duration. Regressors include linear regression, generalized linear regression, decision tree regression, random forest, gradient-boosted trees and deep neural networks. Regressors were first trained on Spark (Databricks platform) with small dataset. Then two best-performed regressors (deep nerual network and gradient-boosted-tree model) were trained on full dataset. 

#### Report
Report explains details
