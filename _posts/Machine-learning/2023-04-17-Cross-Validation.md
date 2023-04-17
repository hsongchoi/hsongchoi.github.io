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
- We are interested in how well our model can predict new data that was not trained.

````python
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

iris_dataset = load_iris()

# Keys
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# Feature_names: A list of the descriptions of features
print("Feature names: \n{}".format(iris_dataset['feature_names'])) 

# Shape of the data array (n: 150, p: 4)
print("Shape of data: {}".format(iris_dataset['data'].shape))

# Target: 0 means setosa, 1 means versicolor, and 2 means virginica.
print("Target:\n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size = 0.2, random_state=12) #A fixed seed
````

- Before splitting, the train_test_split function shuffles the dataset using a pseudorandom number generator.
- It extracts 75% of the rows in the data as the training set and the remaining 25% of the data as the test set.

