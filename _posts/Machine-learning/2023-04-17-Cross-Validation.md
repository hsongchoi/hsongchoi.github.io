---
title:  "Cross-Validation"
categories: ML
tag: [Scikit-learn, Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: true

---

To evaluate our supervised models, we have split our dataset into a training set and a test set, fitted a model on the training set, and evaluated it on the test set using the scoring method.
- We are interested in how well our model can predict new data that was not trained.

```python
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
```


- Before splitting, the train_test_split function shuffles the dataset using a pseudorandom number generator.
- It extracts 75% of the rows in the data as the training set and the remaining 25% of the data as the test set.

# Cross-Validation

- Cross-validation is a statistical method of evaluating generalization performance that is more stable and thorough than using a split into a training and a test set.
  - When performing five-fold cross-validation, the data is first partitioned into five parts of (approximately) equal size, called folds.
  - The first model is trained using the first fold as the test set, and the remaining folds (2–5) are used as the training set.
  - The model is built using the data in folds 2–5; the accuracy is evaluated on fold 1.
- Benefits? 
  - When using cross-validation, each example will be in the test set exactly once: each example is in one of the folds, and each fold is the test set once. On the other hand, train_test_split performs a random split of the data.
  - We use our data more effectively. When using 10-fold cross-validation, we can use nine-tenths of the data (90%) to fit the model. More data will usually result in more accurate models.
- Disadvantage of cross-validation?
  - Increased computational cost: Roughly $k$ times slower than doing a single split of the data.

```python
```

