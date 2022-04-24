# SC1015 Mini-Project
This is for NTU's SC1015 Introduction to Data Science and Artificial Intelligence Mini-Project submission.

SC16 Team 4 Members:
1. Anika Tan Yan Yue
2. Dayna Chia Ching Ning
3. Leo Zhi Kai
---

![Screenshot (52)](https://user-images.githubusercontent.com/97502167/164910694-c357768c-a638-4c67-88b6-33c42366700d.png)

---

## Dataset
The dataset used was a [Video Games Sales Dataset](https://www.kaggle.com/datasets/sidtwr/videogames-sales-dataset) from Kaggle. Our group only used the "Video_Games_Sales_as_at_22_Dec_2016.csv" file.

---

## About
> Video game plays a significant role in modern day entertainment. In 2021, the gaming market is valued at USD 198.4 billion. It is expected to reach USD 339.9 billion by 2027. The growing market continues to attract more companies and individuals every year.

---

## Problem Statement
> What would be the predicted sales of a sequel based on features of the first game?

---


## Why?
> More players join a growing market as it is expected to yield high returns. As such, the success of every game developed and published is of utmost importance. With an accurate estimation of potential sequel game sales, companies are in a better position to do cost-benefit analysis and decide if a sequel game is worth creating.

---

## Libraries Used
- CSV
- Numpy
- Pandas
- Matplotlib
- Scipy
- scikit-learn
- XGBoost

---

## Data Cleaning
1. Drop columns we do not intend to use (Platform, Developer)
2. Remove rows with missing data (e.g. missing genre, publisher etc.)
3. Stripping leading and trailing whitespaces to standardise naming format
4. Use _groupby_ to combine the same game across different platforms into a single row
    - Summing global and regional sales across platforms
    - Averaging _critic_score_, _critic_count_ and _user_score_
5. String comparison to identify first games and their sequels
6. Split 1st and 2nd games into separate dataframes with their corresponding variables then concatenate matching games into a single row in a single dataframe

---

## Exploratory Data Analysis (EDA)
1. Numerical EDA
    - Heatmap for correlation values with Sales2
    - Scatter plots for Critic_Score1, Critic_Count1, User_Score1, NA_Sales1, EU_Sales1, JP_Sales1 and Other_Sales1 with Sales2
3. Categorical EDA
    - Categorical plot for genre count and rating count
    - Boxplot for genre and rating

---

## Machine Learning Models
- One-Hot-Encoding for categorical variable genre
### First Model: sklearn Random Forest Regressor
- With default hyperparameter values
    - Train Explained Variance: 0.835
    - Test Explained Variance: 0.536
- With tuning using GridSearchCV
    - Hyperparameters tuned: n_estimators, max_depth, min_samples_split, min_samples_leaf
    - Train Explained Variance: 0.615
    - Test Explained Variance: 0.601
### Second Model: XGBoost Regressor
- With default hyperparameter values
    - Train Explained Variance: 0.998
    - Test Explained Variance: 0.298
- With tuning using early stopping and RandomizedSearchCV
    - Early stopping with root mean squared error evaluation metric
    - Hyperparameters tuned: learning_rate, max_depth, min_child_weight, gamma, colsample_bytree
    - Train Explained Variance: 0.676
    - Test Explained Variance: 0.660

---

## Conclusion
### Data Insights:
- Sales of the first game has a moderate correlation to the sales of its sequel, so if a certain game sells well, its sequel is likely to sell well too.
- Sales of a sequel game varies according to its genre, in which games of the shooter genre are likely to sell better and games of the adventure genre are likely to do worse.
### Outcome:
For both training and test sets, the tuned XGBoost model has higher explained variance and lower RMSE values. Thus, we conclude that the XGBoost Regressor can predict Sales2 values better in this case, and can reasonably predict the sales of the sequel from the first game’s information with an explained variance of around 0.66 and RMSE of 1.5.
### Recommendations:
- Model can be used to decide if a game should have a sequel
- Publishers can use the model to prioritise order of game development and resource allocation to maximise sales figures
- Forecasted sales figures can attract investors

---

## Presentation Slides
Presentation slides can be found [here](https://docs.google.com/presentation/d/1YAc6b51vfsFI3srWbiPJVsZfXWtp08Otuy1LkDFAc7w/edit?usp=sharing).

---

## Contributors
1. Anika: Data cleaning, EDA, machine learning model, presentation slides, script
2. Dayna: Data cleaning, EDA, machine learning model, presentation slides, script, presenter
3. Zhi Kai: Data cleaning, EDA, machine learning model, presentation slides, script, presenter

---

## References
1. https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
2. https://dataindependent.com/pandas/pandas-get-dummies-pd-get_dummies/
3. https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
4. https://stackabuse.com/random-forest-algorithm-with-python-and-scikit-learn/
5. https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
6. https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/
7. https://mljar.com/blog/xgboost-early-stopping/
8. https://towardsdatascience.com/fine-tuning-xgboost-in-python-like-a-boss-b4543ed8b1e
