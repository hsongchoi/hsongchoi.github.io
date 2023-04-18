---
title:  "Cross-Validation"
categories: ML
tag: [Scikit-learn, Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


To evaluate our supervised models, we have split our dataset into a training set and a test set, fitted a model on **the training set**, and evaluated it on **the test set** using the scoring method.

- We are interested in how well our model can predict new data that was not trained.



```python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

iris_dataset = load_iris()

# Keys
print("Keys of iris_dataset: \n{}".format(iris_dataset.keys()))

# Feature_names: A list of the descriptions of features
print("Feature names: \n{}".format(iris_dataset['feature_names'])) 

# Shape of the data array X(n: 150, p: 4)
print("Shape of data: {}".format(iris_dataset['data'].shape))

# Target y: 0 means setosa, 1 means versicolor, and 2 means virginica.
print("Target:\n{}".format(iris_dataset['target']))

X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], test_size = 0.2, random_state=12) #A fixed seed
```

<pre>
Keys of iris_dataset: 
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
Feature names: 
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
Shape of data: (150, 4)
Target:
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
</pre>
- Before splitting, the train_test_split function shuffles the dataset using a pseudorandom number generator.

- It extracts 75% of the rows in the data as the training set and the remaining 25% of the data as the test set.



```python
# Instantiate a model and fit it to the training set
logreg = LogisticRegression().fit(X_train, y_train)
# evaluate the model on the test set
print("Test set score: {:.2f}".format(logreg.score(X_test, y_test)))
```

<pre>
Test set score: 0.97
</pre>
#  Cross-Validation


- `Cross-validation` is a statistical method of evaluating generalization performance that is more stable and thorough than using a split into a training and a test set.

  - When performing five-fold cross-validation, the data is first partitioned into five parts of (approximately) equal size, called folds.

  - The first model is trained using the first fold as the test set, and the remaining folds (2–5) are used as the training set.

  - The model is built using the data in folds 2–5; the accuracy is evaluated on fold 1.

- **Benefits** 

  - When using cross-validation, each example will be in the test set exactly once: each example is in one of the folds, and each fold is the test set once. On the other hand, train_test_split performs a random split of the data.

  - We use our data more effectively. When using 10-fold cross-validation, we can use nine-tenths of the data (90%) to fit the model. More data will usually result in more accurate models.

- **Disadvantage**

  - Increased computational cost: Roughly $k$ times slower than doing a single split of the data.



```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

iris_dataset = load_iris()
logreg = LogisticRegression()
scores = cross_val_score(logreg, iris.data, iris.target, cv= 5) #Function, X, y
print("Cross-validation scores: {}".format( np.round(scores, 3)))
# Compute the mean to summarize the cross-validation accuracy.
print("Average cross-validation score: {:.2f}".format(scores.mean()))
```

<pre>
Cross-validation scores: [0.967 1.    0.933 0.967 1.   ]
Average cross-validation score: 0.97
</pre>
- Using the mean cross-validation we can conclude that we expect the model to be around 97% accurate on average.

- There is a relatively high variance in the accuracy between folds, ranging from 100% accuracy to 93% accuracy.

    - The model is very dependent on the particular folds used for training because of the small size of the dataset.


# Stratified k-fold Cross-Validation

- Splitting the dataset into k folds by starting with the first one-k-th part of the data might not always be a good idea.

    - What if the test set would be only class 0, and the training set would be only classes 1 and 2? As the classes in training and test sets would be different for all three splits, the three-fold cross-validation accuracy would be zero on this dataset.

    - If 90% of your samples belong to class A and 10% of your samples belong to class B, then stratified cross-validation ensures that in each fold, 90% of samples belong to class A and 10% of samples belong to class B.

- Scikit-learn does not use it for classification, but rather uses **stratified k-fold cross-validation**.

- For regression, Scikit-learn uses the `standard k-fold` cross-validation by default.


# Cross-validation with groups

- When there are groups in the data that are highly related, it is commonly used.

- `GroupKFold` is a variation of k-fold which ensures that the same group is not represented in both testing and training sets.

    - Say you want to build a system to recognize emotions from pictures of faces, and you collect a dataset of pictures of 100 people where each person is captured multiple times, showing various emotions.

    - It is likely that pictures of the same person will be in both the training and the test set.

    - To accurately evaluate the generalization to new faces, we must therefore ensure that the training and test sets contain images of different people.




```python
from sklearn.model_selection import GroupKFold

X = [0.1, 0.2, 2.2, 2.4, 2.3, 4.55, 5.8, 8.8, 9, 10]
y = ["a", "b", "b", "b", "c", "c", "c", "d", "d", "d"]
groups = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

gkf = GroupKFold(n_splits=3)
for train, test in gkf.split(X, y, groups=groups):
    print("%s %s" % (train, test))
```

<pre>
[0 1 2 3 4 5] [6 7 8 9]
[0 1 2 6 7 8 9] [3 4 5]
[3 4 5 6 7 8 9] [0 1 2]
</pre>
# Grid Search

For a better estimate of the generalization performance, instead of using a single split into a training and a validation set, we can use cross-validation to evaluate the performance of each parameter combination.




```python
import pandas as pd
import mglearn
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

iris_dataset = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris_dataset.data, iris_dataset.target, test_size = 0.2, random_state= 12)

dtree = DecisionTreeClassifier()

#Specify the parameters you want to search over using a dictionary.
param_grid = {'max_depth':[1, 2, 3], 'min_samples_split':[2, 3, 6]}

#Instantiate the GridSearchCV class with the model, the parameter grid, and cv strategy.
grid_dtree = GridSearchCV(dtree, param_grid, cv=5, refit = True) #refit = True: Default

#Fit a new model on the training dataset with the parameters.
grid_dtree.fit(X_train, y_train)

results_df = pd.DataFrame(grid_dtree.cv_results_)
print(results_df[ ['params', 'mean_test_score', 'rank_test_score'] ] )

scores = np.array(results_df.mean_test_score).reshape(3, 3)

# Plot the mean cross-validation scores
mglearn.tools.heatmap(scores, xlabel='max_depth', xticklabels=param_grid['max_depth'],
ylabel='min_samples_split', yticklabels=param_grid['min_samples_split'], cmap="viridis")
```

<pre>
                                     params  mean_test_score  rank_test_score
0  {'max_depth': 1, 'min_samples_split': 2}         0.625000                7
1  {'max_depth': 1, 'min_samples_split': 3}         0.625000                7
2  {'max_depth': 1, 'min_samples_split': 6}         0.625000                7
3  {'max_depth': 2, 'min_samples_split': 2}         0.908333                4
4  {'max_depth': 2, 'min_samples_split': 3}         0.908333                4
5  {'max_depth': 2, 'min_samples_split': 6}         0.908333                4
6  {'max_depth': 3, 'min_samples_split': 2}         0.950000                1
7  {'max_depth': 3, 'min_samples_split': 3}         0.950000                1
8  {'max_depth': 3, 'min_samples_split': 6}         0.950000                1
</pre>
<pre>
<matplotlib.collections.PolyCollection at 0x7fbc1a55d0a0>
</pre>
<img src="data:image/png;base64,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K1bohhtuOO+iAAD4JZcea3/xxRfrySefVGpqqhISEiRJmzZt0saNG3Xffffp+eefd/QdP368eyoFAHgtl45ZxcbGVm/lNpv27dt3zkVVF8esAKBuq+4xK5e2rLiRLQCgNrnlouCysjJt27ZNJ06ccMfqAABw4lJY3XPPPVqwYIGkM0F15ZVXqkuXLoqJiVFqaqo76wMAwLWw+te//qWOHTtKkt577z3t379f3377re655x49/PDDbi0QAACXwionJ0dNmjSRJL3//vtKSkpSmzZtNHLkyN983D0AAK5wKawiIyP1zTffqKysTGvWrNE111wjSTp58qR8fTkrDwDgXi6dDXjbbbdpyJAhioqKks1mU9++fSVJX3zxhS666CK3FggAgEth9dhjjykuLk6ZmZlKSkqSv7+/JMnX11cPPfSQWwsEAMClsJKkG2+8sUJbcnKy03SHDh30/vvvKyYmxtW3AQCgZh++uH//fpWUlNTkWwAAvABPCgYAGI+wAgAYj7ACABiPsAIAGI+wAgAYz21hlZubW6Ft7ty5ioyMdNdbAAC8lEthNX36dC1btswxPWTIEDVq1EhNmzbV9u3bHe3Dhg1TgwYNzr9KAIBXcyms5s6d67jQd+3atVq7dq1Wr16txMRE3X///W4tEAAAl+5gkZWV5QirVatWaciQIerXr59atmypbt26ubVAAABc2rJq2LChMjMzJcnpruuWZamsrMx91QEAIBe3rAYPHqxhw4bpD3/4g44dO6bExERJ0rZt29S6dWu3FggAgEth9eyzz6ply5bKzMzUjBkzFBwcLOnM7sHRo0e7tUAAAGyWZVmeLsJV+fn5stvtWrS1o4JCeOgjANQ1JwvKNLzzduXl5Sk0NLTKfi5fZ7V48WL17NlT0dHR+u9//ytJeu6557RixQpXVwkAQKVcCquUlBRNmDBBiYmJys3NdZxUERYWpueee86d9QEA4FpYzZ49W/Pnz9fDDz8sX9+fd7917dpVO3fudFtxAABILoZVRkaGOnfuXKHd399fhYWF510UAAC/5FJYxcbGatu2bRXaV69erXbt2p1vTQAAOHHp1PX7779fY8aMUVFRkSzLUlpamt58801NmzZNL7/8srtrBAB4OZfC6rbbblNpaakeeOABnTx5UsOGDVPTpk01a9Ys/eUvf3F3jQAAL+dSWEnSqFGjNGrU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bt36N9e1adMm3XrrrU7TnTt3liT5+flJOnPCCGAqwgpwUevWrfXaa6/pgw8+UGxsrBYvXqwvv/zScXadJL3wwgv6/PPPtWPHDsXExGj16tW6+eab9cUXXzhCoirjx49Xjx49NGPGDA0aNEgffvih1qxZ49TnkUce0fXXX6+YmBglJSXJx8dHO3bs0M6dO/WPf/zD0e/tt99W165d1bNnT73xxhtKS0vTggULJEkREREKDAzUmjVr1KxZMwUEBMhut7vxNwWcP45ZAS668847NXjwYA0dOlTdunXTsWPHnLayvv32W91///168cUXFRMTI+lMeOXm5urvf//7WdffvXt3vfzyy5o9e7Y6deqkDz/8UJMnT3bqc+2112rVqlVau3atLr30UnXv3l0zZ85UixYtnPpNmTJFS5cuVXx8vF599VW98cYbjq27evXq6fnnn9fcuXMVHR2tgQMHnu+vBnA7m1WdHeMA6iybzably5dz6yTUaWxZAQCMR1gBHpKYmOh0yvkvX1OnTvV0eYBR2A0IeMihQ4d06tSpSueFh4crPDy8lisCzEVYAQCMx25AAIDxCCsAgPEIKwCA8QgrAIDxCCsAgPEIKwCA8QgrAIDxCCsAgPH+PwxFBDWimP9SAAAAAElFTkSuQmCC"/>

- The `min_samples_split` parameter is searching over interesting values but the `max_dept` parameter is not—or it could mean the `max_dept` parameter is not important.



```python
#To evaluate how well the best found parameters generalize, we can call score on the test set.
print("Best parameters: {}".format(grid_dtree.best_params_))
print("Best cross-validation score: {:.2f}".format(grid_dtree.best_score_))
print("Test set score: {:.2f}".format(grid_dtree.score(X_test, y_test)))
```

<pre>
Best parameters: {'max_depth': 3, 'min_samples_split': 2}
Best cross-validation score: 0.95
Test set score: 0.93
</pre>