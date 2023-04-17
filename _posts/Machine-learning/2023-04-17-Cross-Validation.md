---
title:  "Cross-Validation"
categories: Machine Learning
tag: [Scikit-learn, Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: false

---

To evaluate our supervised models, we have split our dataset into a training set and a test set, fitted a model on the training set, and evaluated it on the test set using the scoring method.
- We are interested in how well our model can make predictions for new data that was not trained.

````Py
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

# Format: Similar to a dictionary
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# Feature_names: A list of the descriptions of features
print("Feature names: \n{}".format(iris_dataset['feature_names'])) 

# 
````



