This repository contains data science projects completed by me for academic, self-learning, and hobby purposes. Presented in the form of iPython notebooks in Python and R. More projects on broader ML/DL to be added.

_Note: Data used in the projects (accessed under data directory) is for demonstration purposes only._

## Contents

- ### Machine Learning

  - [Supervised Learning: Iowa House Prices](https://github.com/yl5787/data-science-projects/blob/main/house-prices/house-prices.ipynb): Testing out several different advanced regression algorithms (Ridge, Lasso, Elastic Net, Random Forest, Gradient Boosting) to build a model that predicts house prices in Iowa based on 79 explanatory variables describing aspects of the houses. Determined which regression model best predicted house prices RMSE and R-squared.

  - [Supervised Learning: Titanic Survivors](https://github.com/yl5787/data-science-projects/blob/main/titanic/titanic.ipynb): Testing out several different classification algorithms (Random Forest, K-Nearest Neighbors, Decision Tree, Support Vector Machine) to build a model that predicts what sort of people were more likely to survive the Titanic. F1-score for model evaluation.

  - [Predict Future Sales](https://github.com/yl5787/data-science-projects/blob/main/predicting-sales/predicting-sales.ipynb): Predicting total sales of every product and store in the next month based off a time-series dataset consisting of daily sales data. Feature engineering process includes producing Previous Value Benchmark and clipping true target values into [0,20] range as recommended by the author. To be updated.

  _Tools: pandas, Matplotlib, seaborn, NumPy, SciPy, scikit-learn_

- ### Natural Language Processing

  - [Disaster Tweet Prediction](https://github.com/yl5787/data-science-projects/blob/main/disaster-tweets/disaster-tweets.ipynb): A model to predict which Tweets are about real disasters and which aren’t based on a dataset of 10,000 tweets using TF-IDF Vectorizer and Ridge Classifier for modeling and F1 score for evaluation. Exploratory data analysis and wordcloud visualization of tweets.

  _Tools: pandas, NumPy, Matplotlib, seaborn, scikit-learn, wordcloud, NLTK, PIL_

- ### Data Analysis and Visualization

  - [World Bank Science and Technology Data Analysis](https://github.com/yl5787/data-science-projects/blob/main/world-bank-science-technology/world-bank-science-technology.ipynb): Exploratory and descriptive analysis, visualization, and statistical tests including t-tests and correlation

  _Tools: pandas, Matplotlib, seaborn, SciPy, NumPy_
  
- ### Demonstration

  - [Categorical Data Encoding](https://github.com/yl5787/data-science-projects/blob/main/encodings-for-categorical-data/encoding-demonstrations.ipynb): Three encoding methods (ordinal, one-hot, dummy) performed on the breast cancer dataset that classifies patient data as a recurrence or no recurrence of cancer. Uses logistic regression for prediction.

  _Tools: pandas, NumPy, scikit-learn_
