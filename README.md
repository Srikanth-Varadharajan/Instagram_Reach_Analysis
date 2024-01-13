**Instagram Reach Analysis**<br>
_This repository contains Python code for analyzing Instagram reach using a dataset. The dataset includes information about various factors affecting post impressions on Instagram._

Code Explanation:
Importing Libraries:

import pandas as pd<br>
import numpy as np<br>
import matplotlib.pyplot as plt<br>
import seaborn as sns<br>
import plotly.express as px<br>
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator<br>
from sklearn.model_selection import train_test_split<br>
from sklearn.linear_model import PassiveAggressiveRegressor<br>
_The code begins by importing necessary libraries for data manipulation, visualization, and machine learning.
_<br>
**Loading and Exploring Data:**<br>

The dataset is loaded from a CSV file, and basic exploratory data analysis is performed, including checking for missing values and displaying summary statistics.<br>

**Data Visualization:**<br>

The code uses Matplotlib and Seaborn to create distribution plots for impressions from different sources (Home, Hashtags, Explore).<br>
A pie chart is generated to visualize the distribution of impressions from different sources.<br>
A word cloud is created based on Instagram post captions.<br>
**Analyzing Relationships:**<br>

Scatter plots are created to analyze relationships between the number of likes, comments, shares, saves, profile visits, and follows with impressions.<br>
**Correlation Analysis:**<br>


The correlation of each column with the 'Impressions' column is calculated and displayed.<br>


**Linear Regression Model:**<br>


x = np.array(df[['Likes', 'Shares', 'Saves', 'Comments', 'Profile Visits', 'Follows']])<br>
y = np.array(df['Impressions'])<br>
X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size=0.1, random_state=225)<br>
model = PassiveAggressiveRegressor()<br>
model.fit(X_train, Y_train)<br>
model.score(X_test, Y_test)<br>
**Prediction:**<br>


features = np.array([[127, 2, 90, 4, 32, 10]])<br>
model.predict(features)<br>
A Passive Aggressive Regressor model is trained and tested on the dataset, and a prediction is made based on specific feature values.<br>
