# Machine Learning Modeling and Prediction of Housing Prices (In Progress)

## Aim

The goal of this study is to accurately predict housing prices by comparing various machine learning models: linear, ridge, lasso, decision tree, random forest, gradient boosting, and convolutional neural network (CNN)-based regression and classification.  This dataset includes relevant features and images of houses in Southern California (https://www.kaggle.com/ted8080/house-prices-and-images-socal).


## Exploratory Data Analysis

Figure.  Histograms.

![](figure/histograms.png)


Figure.  Scatter plots.

![](figure/scatterplots.png)


Figure.  Correlation matrix.

![](figure/correlation_matrix.png)


## Results

Figure.  Residual plots.

![](figure/residual_rf.png)

![](figure/residual_dt.png)

![](figure/residual_gbr.png)

![](figure/residual_ridge.png)

![](figure/residual_lasso.png)

![](figure/residual_linear.png)


Figure.  Gradient boosting regression: training set: RMSE vs learing rate at specific estimators.

![](figure/train_rmse_lr.png)


Figure.  Gradient boosting regression: test set: RMSE vs learing rate at specific estimators.

![](figure/test_rmse_lr.png)


Figure.  Gradient boosting regression: permutation importances of training set.

![](figure/permutation.png)


Table.  Overall training and test results.

![](figure/table.jpg)
