# SC1015 Mini-Project
This is for NTU's SC1015 Introduction to Data Science and Artificial Intelligence Mini-Project submission.

Team 4 Members:
1. Anika Tan Yan Yue
2. Dayna Chia Ching Ning
3. Leo Zhi Kai

## Dataset
The dataset used was a [Video Games Sales Dataset](https://www.kaggle.com/datasets/sidtwr/videogames-sales-dataset) from Kaggle. Our group only used the "Video_Games_Sales_as_at_22_Dec_2016.csv" file.

## Problem Statement
> What would be the predicted sales of a sequel based on features of the first game?

## Libraries Used
- CSV
- Numpy
- Pandas
- Matplotlib
- Scipy
- scikit-learn
- XGBoost

## Data Cleaning
1. Drop columns we do not intend to use (Platform, Developer)
2. Remove rows with missing data (e.g. missing genre, publisher etc.)
3. Stripping leading and trailing whitespaces to standardise naming format
4. Use _groupby_ to combine the same game across different platforms into a single row
    - Summing global and regional sales across platforms
    - Averaging _critic_score_, _critic_count_ and _user_score_
5. String comparison to identify sequel games
6. Split 1st and 2nd games into separate dataframes with their corresponding variables then concatenate matching games into a single row in a single dataframe

## Exploratory Data Analysis (EDA)
1. Numerical EDA
    - Heatmap for correlation values with Sales2
    - Scatter plots for Critic_Score1, Critic_Count1, User_Score1, NA_Sales1, EU_Sales1, JP_Sales1 and Other_Sales1 with Sales2
3. Categorical EDA
    - Categorical plot for genre count and rating count
    - Boxplot for genre and rating

## Machine Learning Model
- One-Hot-Encoding for categorical variable genre
1. First model
    - Random Forest Regressor
        - Train Explained Variance: 0.835
        - Test Explained Variance: 0.536
    - Random Forest Regressor with GridSearchCV
        - Hyperparameters: n_estimators, max_depth, min_samples_split, min_samples_leaf
        - Train Explained Variance: 0.61
        - Test Explained Variance: 0.60
2. Second model
    - XGBoost Regression
        - Train Explained Variance: 0.99
        - Test Explained Variance: 0.30
    - XGBoost with RandomizedSearchCV
        - Early stopping with root mean squared error evaluation metric
        - Hyperparameters: learning_rate, max_depth, min_child_weight, gamma, colsample_bytree
        - Train Explained Variance: 0.68
        - Test Explained Variance: 0.66

## Conclusion
- The XGBoost with RandomizedSearchCV model performed the best

## Contributors
1. Anika: Data cleaning, EDA, machine learning model, presentation slides, script
2. Dayna: Data clearning, EDA, machine learning model, presentation slides, script, presenter
3. Zhi Kai: Data clearning, EDA, machine learning model, presentation slides, script, presenter

## References
1. https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
2. https://dataindependent.com/pandas/pandas-get-dummies-pd-get_dummies/
3. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
4. https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
5. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
6. https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
7. https://mljar.com/blog/xgboost-early-stopping/
8. https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
