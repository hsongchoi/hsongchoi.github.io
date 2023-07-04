---
title:  "Data Preprocessing"
categories: ML
tag: [Scikit-learn, Python, Machine-Learning, Data-Preprocessing]
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

# Encoding for Categorical variables

- Most Machine learning algorithms can not handle categorical variables unless we convert them to numerical values.
-  Feature engineering is essential, yet it is also arguably one of the most manually intensive steps in the applied ML process.
-  Handling non-numeric data is a critical component of nearly every machine-learning process. 
- Many algorithms’ performances vary based on how Categorical variables are encoded.
- Categorical variables can be divided into two categories: Nominal (No particular order: Red, Yellow, Pink, Blue) and Ordinal (some ordered: Red, Yellow, Pink, Blue).
- My recommendation will be to try each of these with the smaller datasets and then decide where to focus on tuning the encoding process.

> We need to use the mapping values created at the time of training. 
>
> This process is the same as scaling or normalization, where we use the train data to scale or normalize the test data. 
>
> Then map and use the same value in testing time pre-processing. 
>
> We can even create a dictionary for each category and map the value and then use the dictionary at testing time.

![image-20230703221933020](/images/2023-04-19-Data Preprocessing/image-20230703221933020.png)

![image-20230703221949599](/images/2023-04-19-Data Preprocessing/image-20230703221949599.png)

## One Hot encoding

By far, the most common way to represent categorical variables is using the `one-hot encoding` or one-out-of-N encoding, also known as `dummy variables`.

- We map each category to a vector that contains 1 and 0, denoting the presence or absence of the feature.
- The number of vectors depends on the number of categories for features.
- Cons: This method produces many columns that **slow down the learning significantly** if the number of the category is very high for the feature.



```python
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
items = ['TV', 'Fridge', 'Microwave', 'Computer', 'Fan', 'Fan', 'Blender', 'Blender']

le = LabelEncoder()
le.fit(items)
labels = le.transform(items).reshape(-1, 1)

print(le.transform(items))

ohe = OneHotEncoder()
ohe.fit(labels)
ohe_labels = ohe.transform(labels)
print(ohe_labels.toarray())
print(ohe_labels.shape)
```

<pre>
[5 3 4 1 2 2 0 0]
[[0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 1. 0.]
 [0. 1. 0. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]
 [1. 0. 0. 0. 0. 0.]]
(8, 6)
</pre>
- When working with data that was input by humans (say, users on a website), there might not be a fixed set of categories, and differences in spelling and capitalization might require preprocessing.

- Check the contents of a column.



```python
pd.DataFrame(items).value_counts()
```

<pre>
Blender      2
Fan          2
Computer     1
Fridge       1
Microwave    1
TV           1
dtype: int64
</pre>
> We can represent all categories by N-1 (N= No of Category) as sufficient to encode the one that is not included.
>
> - Usually, for Regression, we use N-1 (drop the first or last column of One Hot Coded new feature ). 
>   - The linear Regression has access to all of the features as it is being trained and therefore examines the whole set of dummy variables altogether.
>   - This means that N-1 binary variables give complete information about (represent completely) the original categorical variable to the linear Regression. 
> - Still, for classification, the recommendation is to use all N columns, as most of the tree-based algorithm builds a tree based on all available variables.

### The get_dummies function

`The get_dummies function` automatically transforms all columns that have object type (like strings) or are categorical.

- Pandas has **get_dummies** function, which is quite easy to use. 
- Using get_dummies will only encode the string feature and will not change the integer feature.
- If you want dummy variables to be created for the “Integer Feature” column, you can explicitly list the columns you want to encode using the columns parameter. Then, both features will be treated as categorical.

    - pd.get_dummies(demo_df, columns=['Integer Feature', 'Categorical Feature']).



```python
import pandas as pd
pd.get_dummies(items)
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Blender</th>
      <th>Computer</th>
      <th>Fan</th>
      <th>Fridge</th>
      <th>Microwave</th>
      <th>TV</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

## Label Encoding

- In this encoding, each category is assigned **a value from 1 through N** (where N is the number of categories for the feature. 
- One major issue with this approach is there is no relation or order between these classes, but the algorithm might **consider them as some order or some relationship.**
- In below example it may look like (Cold<Hot<Very Hot<Warm….0 < 1 < 2 < 3 ).

![image-20230703214037515](/images/2023-04-19-Data Preprocessing/image-20230703214037515.png)

- Pandas **factorize** also perform the same function.

![image-20230703214102936](/images/2023-04-19-Data Preprocessing/image-20230703214102936.png)

## Ordinal Encoding

- This is reasonable only for ordinal variables.
- We do Ordinal encoding to ensure the encoding of variables retains the ordinal nature of the variable.
- slightly different as **Label coding**
  - as per the order of data (Pandas assigned Hot (0), Cold (1), “Very Hot” (2), and Warm (3)) or
  - as per alphabetically sorted order (scikit-learn assigned Cold(0), Hot(1), “Very Hot” (2), and Warm (3)).
- If we consider the temperature scale as the order, then the ordinal value should from cold to “Very Hot. “ Ordinal encoding will assign values as ( Cold(1) <Warm(2)<Hot(3)<”Very Hot(4)). Usually, Ordinal Encoding is done starting from 1.

![image-20230703215127525](/images/2023-04-19-Data Preprocessing/image-20230703215127525.png)

## Helmert Encoding

- In this encoding, the mean of the dependent variable for a level is compared to the mean of the dependent variable over all previous levels.

![image-20230703215519786](/images/2023-04-19-Data Preprocessing/image-20230703215519786.png)

## Binary Encoding

- Binary encoding converts a category into binary digits.
- Each binary digit creates one feature column.
- If there are **n** unique categories, then binary encoding results in only $log_2(n)$ features. 
  - For 100 categories, One Hot Encoding will have 100 features, while Binary encoding will need just seven features.

![image-20230703215859547](/images/2023-04-19-Data Preprocessing/image-20230703215859547.png)

![image-20230703215910010](/images/2023-04-19-Data Preprocessing/image-20230703215910010.png)

## Frequency Encoding

- It is a way to utilize the frequency of the categories as labels.
- In the cases where the frequency is related somewhat to the target variable, it helps the model understand and assign the weight in direct and inverse proportion, depending on the nature of the data.

![image-20230703220030806](/images/2023-04-19-Data Preprocessing/image-20230703220030806.png)

## Mean Encoding

- Mean Encoding or Target Encoding is one viral encoding approach followed by Kagglers. 
- There are many variations of this.
- Mean encoding is similar to label encoding, except here, labels are correlated directly with the target.
- This encoding method brings out the relation between similar categories, but the connections are **bounded within the categories and the target itself**.
- Pros: it **does not affect the volume of the data** and helps in faster learning.
- Usually, Mean encoding is **notorious for over-fitting**; thus, a regularization with cross-validation or some other approach is a must on most occasions.

1. Select a categorical variable you would like to transform.
2. Group by the categorical variable and obtain aggregated sum over the “Target” variable. (total number of 1’s for each category in ‘Temperature’)
3. Group by the categorical variable and obtain aggregated count over “Target” variable
4. Divide the step 2 / step 3 results and join it back with the train.

![image-20230703220830470](/images/2023-04-19-Data Preprocessing/image-20230703220830470.png)

![image-20230703220922801](/images/2023-04-19-Data Preprocessing/image-20230703220922801.png)

- Mean encoding can embody the target in the label, whereas label encoding does not correlate with the target.
- In the case of many features, mean encoding could prove to be a much simpler alternative. 
- Mean encoding tends to group the classes, whereas the grouping is random in label encoding.

There are many variations of this target encoding in practice, like smoothing. Smoothing can implement as below:

![image-20230703221019957](/images/2023-04-19-Data Preprocessing/image-20230703221019957.png)

## Summary of Categorical Encoding

![image-20230703214415862](/images/2023-04-19-Data Preprocessing/image-20230703214415862.png)

# Encoding for Quantitive variables

## Binning

Linear models can only model linear relationships, which are lines in the case of a single feature. The decision tree can build a much more complex model of the data. However, this is **strongly dependent on the representation of the data.**

One way to make linear models more powerful on continuous data is to use binning (also known as discretization) of the feature to split it up into multiple features.



```python
import mglearn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

X, y = mglearn.datasets.make_wave(n_samples=100)
line = np.linspace(-3, 3, 1000, endpoint=False).reshape(-1, 1)
reg = DecisionTreeRegressor(min_samples_split=3).fit(X, y)
plt.plot(line, reg.predict(line), label="decision tree")
reg = LinearRegression().fit(X, y)

plt.plot(line, reg.predict(line), label="linear regression")
plt.plot(X[:, 0], y, 'o', c='k')
plt.ylabel("Regression output")
plt.xlabel("Input feature")
plt.legend(loc="best")
```

<pre>
<matplotlib.legend.Legend at 0x7f886b901f40>
</pre>
<img src="data:image/png;base64,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sh49epi8PnDggLNNAwDs27cPe/bswdy5c6VlHMehrKwMpaWlCAkJQUZGBubNm4cjR46gqKgIVVVVKCsrw82bN1GrVi27bejatav0/ytXruDcuXMYP348Hn/8cWl5VVUVateubfexCIIwhEQNQRCKMU56ZRgGWq3W6nYMwwAANBohjY/neem9yspKh9mn1WoxZ84cpKammrwXFBSE//77D3fffTcmTpyI119/HVFRUdi5cyfGjx9v1Q6GYQzsNme7vjASz83y5cvRvXt3g/VYllX8exEEoQwSNQRBOJRdu3Zh7NixBq87deoEAIiJiQEA5OfnIzIyEgBMvDgBAQHgOM6mY3fu3BnHjx9H06ZNZd/fu3cvqqqqsGDBAklgrV+/XtHxY2JikJ+fL70+efIkSktLLdpTt25dxMfH48yZMxgzZozaX4cgCJWQqCEIwqF888036Nq1K3r37o0vv/wSu3fvxooVKwAATZs2RWJiImbPno033ngDJ0+exIIFCwy2T0pKQklJCbZv344OHTogJCQEISEhio49a9Ys3HvvvUhMTMSIESOg0WiQnZ2NgwcP4o033kCTJk1QVVWFDz74APfddx/++OMPfPLJJ4qO379/f3z44Ye4/fbbodVq8cILLygq1549ezamTp2K8PBwDB48GOXl5di7dy+uXbuGmTNnKjyrBEEogUq6CYJwKHPmzMFXX32F9u3bY/Xq1fjyyy/RunVrAEL4at26dTh27Bg6dOiAt956C2+88YbB9j179sTEiRPx4IMPIiYmBm+//bbiYw8cOBA//PADtm3bhm7duuH222/HwoUL0bBhQwBAx44dsXDhQrz11lto27YtvvzyS8yfP1/R8RcsWIDExET06dMHo0ePxrPPPqtIbE2YMAGffvqpVIbet29frFq1Co0aNVL8exEEoQyGNw4SEwRB2AjDMNi0aZNbOhETBEGQp4YgCIIgCJ+ARA1BEARBED4BJQoTBOEwKJpNEIQ7IU8NQRAEQRA+AYkagiAIgiB8AhI1BEEQBEH4BCRqCIIgCILwCUjUEARBEAThE5CoIQiCIAjCJyBRQxAEQRCET0CihiAIgiAIn+D/AevZU7Im5fd0AAAAAElFTkSuQmCC"/>

- We imagine a partition of the input range for the feature (in this case, the numbers from –3 to 3) into a fixed number of bins—say, 10.

- We create 11 entries, which will create 10 bins.



```python
bins = np.linspace(-3, 3, 11)
print("bins: {}".format(bins))
```

<pre>
bins: [-3.  -2.4 -1.8 -1.2 -0.6  0.   0.6  1.2  1.8  2.4  3. ]
</pre>

```python
which_bin = np.digitize(X, bins=bins)
print("\nData points:\n", X[:5])
print("\nBin membership for data points:\n", which_bin[:5])
```

<pre>

Data points:
 [[-0.75275929]
 [ 2.70428584]
 [ 1.39196365]
 [ 0.59195091]
 [-2.06388816]]

Bin membership for data points:
 [[ 4]
 [10]
 [ 8]
 [ 6]
 [ 2]]
</pre>

```python
from sklearn.preprocessing import OneHotEncoder
# transform using the OneHotEncoder
encoder = OneHotEncoder(sparse=False)
# encoder.fit finds the unique values that appear in which_bin
encoder.fit(which_bin)
# transform creates the one-hot encoding
X_binned = encoder.transform(which_bin)
print(X_binned[:5])

#Because we specified 10 bins, the transformed dataset X_binned now is made up of 10 features.
print("X_binned.shape: {}".format(X_binned.shape))
```

<pre>
[[0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
 [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
 [0. 0. 0. 0. 0. 0. 0. 1. 0. 0.]
 [0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
X_binned.shape: (100, 10)
</pre>

```python
line_binned = encoder.transform(np.digitize(line, bins=bins))
reg = LinearRegression().fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='linear regression binned')
reg = DecisionTreeRegressor(min_samples_split=3).fit(X_binned, y)
plt.plot(line, reg.predict(line_binned), label='decision tree binned')
plt.plot(X[:, 0], y, 'o', c='k')
plt.vlines(bins, -3, 3, linewidth=1, alpha=.2)
plt.legend(loc="best")
plt.ylabel("Regression output")
plt.xlabel("Input feature")
```

<pre>
Text(0.5, 0, 'Input feature')
</pre>
<img src="data:image/png;base64,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HmXjdHVMaoiIrOXhU9JiUkU6nYjbxVp4uavg5iZIf24JRAndVDqdDiqVCgcPHoRKZdzyor/Qe3t7S4sLQK1atfDHH38gJSUFW7ZswdSpUzFt2jTs37+/3Ewkc/GJoghBuHsuys6WEgRB0u9mSs2axi1cOp0OTzzxBD788MNy+4aGhuKvv/4CAPz3v/9FWFiY0f2enp5WP3/p38uU0r+rfl+dTmfyfv0++nOh32/58uVGiSIAw2tb2fPmSpjUEBFZSxAkdQFVSCcCblrAXQVITWokatq0KTw8PLB37140aNAAQMnA3BMnThi6Tdq3bw+tVovs7GxERESYPE7btm2xbds2TJ8+XdLzuru746GHHsJDDz2E9957DwEBAdi+fbuhS0SvVatWKC4uxu+//27ofrpy5QpOnDiBli1bVvbXNsxoMjXmp6wOHTogMTER4eHhcHcvfzls1aoVPD09kZ6ebrGryZS9e/eWu92iRQurjmGNevXqISwsDKdPn8bQoUNN7tOqVSt88803uHXrliFZLRunq+NAYSIiBfL19cWIESMwefJkbNu2DX/99Reef/55uLnd/di/5557MHToUAwbNgxJSUlIS0vD/v378eGHH+Knn34CALz55pvYv38/xowZg8OHD+Off/7B8qWfG80Q0vvxxx/x0UcfITU1FWfPnsXXX38NnU5n1OWl16xZM0RHR2PUqFHYvXs3/vzzTzz77LMICwtDdHR0pX/vhg0bQhAE/Pjjj7h06RJu3rxpdt9XXnkFV69exeDBg7Fv3z6cPn0aW7ZswYsvvgitVotatWph0qRJGD9+PL766iucOnUKhw4dwqeffmoYfGvOnj17MHfuXJw4cQKffvopvvvuO4wbN67Sv5cU06ZNw+zZs7F48WKcOHECR44cQXx8PBYuXAgAGDJkCNzc3DBixAgcPXoUP/30E+bPn2/TmOyNSQ0RkULNmzcP3bt3R79+/fDQQw+hW7du5caVxMfHY9iwYZg4cSKaN2+Ofv364ffff4dGowFQkvhs2bIFf/75J/7v//4PXbt2wY8//GCyZSMgIABJSUno2bMnWrZsic8//xyrV69G69atTcYXHx+Pjh074vHHH0fnzp0hiiJ++umnKhXoCwsLw/Tp0/HGG2+gXr16GDt2rNl969evjz179kCr1eKRRx5BmzZtMG7cOPj7+xuSvw8+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"/>

- The dashed line and solid line are exactly on top of each other, meaning the linear regression model and the decision tree make exactly the same predictions. For each bin, they predict a constant value.

- The linear model became much more flexible, because it now has a different value for each bin, while the decision tree model got much less flexible.

- **The linear model benefited greatly in expressiveness from the transformation of the data.**

- If the dataset is very large and high-dimensional, but some features have nonlinear relations, the binning can be a great way to increase modeling power.



> 1. Binning features generally has `no beneficial effect for tree-based models`, as these models can learn to split up the data anywhere. In a sense, that means decision trees can learn whatever binning is most useful

for predicting on this data. 2. decision trees look at multiple features at once, while binning is usually done on a per-feature basis.

# Feature scaling

- In many machine learning algorithms, to bring all features in the same standing, we need to do scaling so that one significant number doesn’t impact the model just because of their large magnitude.
- `Feature scaling` in machine learning is one of the most critical steps during the pre-processing of data before creating a machine learning model.
- Like most other machine learning steps, feature scaling too is a **trial and error process**, not a single silver bullet.
- The most common techniques of feature scaling are `Normalization` and `Standardization`.
  - **Normalization** is used when we want to bound our values between two numbers, typically, between [0,1] or [-1,1].
  - While **Standardization** transforms the data to have zero mean and a variance of 1, they make our data **unitless**.

`Normalizer` does a very different kind of rescaling. It scales each data point such that the feature vector has a Euclidean length of 1. In other words, it projects a data point on the circle (or sphere, in the case of higher dimensions) with a radius of 1.

- SVM, Linear Regression, Logistic Regression assume that data follow the Gaussian distribution.

- It is important to apply exactly `the same transformation to the training set and the test set` for the supervised model to work on the test set.

  - We call fit on the `training set`, and then call transform on the `training and test sets`.

![image-20230704101327253](/images/2023-04-19-Data Preprocessing/image-20230704101327253.png)

## Why need scaling?

The ML algorithm is sensitive to the “**relative scales of features**.

1. The feature with a higher value range dominants the other features.

- If there is a vast difference in the range, it makes the underlying assumption that higher ranging numbers have superiority of some sort.
- These more significant number starts playing a more decisive role while training the model.

![image-20230704000012733](/images/2023-04-19-Data Preprocessing/image-20230704000012733.png)

- These more significant number of weights starts playing a more decisive role while training the model. 
- Interestingly, if we convert the weight to “Kg,” then “Price” becomes dominant.

2. Few algorithms like Neural network gradient descent **converge much faster** with feature scaling than without it.

![image-20230704000242875](/images/2023-04-19-Data Preprocessing/image-20230704000242875.png)

3. **Saturation**, like in the case of sigmoid activation in Neural Network, scaling would help not to saturate too fast.

## When to do scaling?

Rule of thumb we may follow here is an algorithm that computes distance or assumes normality, **scales your features.**

Some examples of algorithms where feature scaling matters are:

- **K-nearest neighbors** (KNN) with a Euclidean distance measure is sensitive to magnitudes and hence should be scaled for all features to weigh in equally.
- **K-Means** uses the Euclidean distance measure here feature scaling matters.
- Scaling is critical while performing **Principal Component Analysis(PCA)**. PCA tries to get the features with maximum variance, and the variance is high for high magnitude features and skews the PCA towards high magnitude features.
- We can speed up **gradient descent** by scaling because θ descends quickly on small ranges and slowly on large ranges, and oscillates inefficiently down to the optimum when the variables are very uneven.

Algorithms that do not require normalization/scaling are the ones that **rely on rules**. They would not be affected by any monotonic transformations of the variables. Scaling is a monotonic transformation. 

- Examples of algorithms in this category are all the tree-based algorithms — **CART, Random Forests, Gradient Boosted Decision Trees**. These algorithms utilize rules (series of inequalities) and **do not require normalization**.

Algorithms like **Linear Discriminant Analysis(LDA), Naive Bayes is** by design equipped to handle this and give weights to the features accordingly. Performing features scaling in these algorithms may not have much effect.

### MinMaxScaler (Normalization)

- All of the features are between 0 and 1. 
- This Scaler shrinks the data within the range of -1 to 1 if there are negative values. 
- This Scaler responds well if the standard deviation is small and when a distribution is **not Gaussian**.
- This Scaler is **sensitive to outliers**.

`MinMaxScaler` shifts the data such that all features are exactly between 0 and 1. For the two-dimensional dataset this means all of the data is contained within the rectangle created by the x-axis between 0 and 1 and the y-axis

between 0 and 1.

$$
X_{std} = \frac{X - X_{min}}{X_{max} - X_{min}},
$$
and
$$
X_{scaled} = X_{std} * (max - min) + min.
$$

where max, min represent feature ranges.

```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df)

iris_df_scaled = pd.DataFrame(data = iris_scaled, columns =iris.feature_names)
print('Mins of the features:\n{}'.format(iris_df_scaled.min())) 
print('Maxes of the features:\n{}'.format(iris_df_scaled.max())) 
```

<pre>
Mins of the features:
sepal length (cm)    0.0
sepal width (cm)     0.0
petal length (cm)    0.0
petal width (cm)     0.0
dtype: float64
Maxes of the features:
sepal length (cm)    1.0
sepal width (cm)     1.0
petal length (cm)    1.0
petal width (cm)     1.0
dtype: float64
</pre>


```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()

from sklearn.svm import SVC
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, random_state=23)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test set accuracy: {:.2f}".format(svm.score(X_test, y_test)))
```

<pre>
Test set accuracy: 0.94
</pre>


```python
# Preprocessing using 0-1 scaling

scaler = MinMaxScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# learning an SVM on the scaled training data
svm.fit(X_train_scaled, y_train)

# scoring on the scaled test set/ The effect of scaling the data is quite significant.
print("Scaled test set accuracy: {:.2f}".format(svm.score(X_test_scaled, y_test)))
```

<pre>
Scaled test set accuracy: 0.97
</pre>

### Unit Vector Scaler (Normalization)

![image-20230704003120987](/images/2023-04-19-Data Preprocessing/image-20230704003120987.png)

- Scaling is done considering the whole feature vector to be of unit length.
- Scaling to unit length shrinks/stretches a vector (a row of data can be viewed as a *D*-dimensional vector) to a unit sphere. When used on the entire dataset, the transformed data can be visualized as a bunch of vectors with different directions on the *D*-dimensional unit sphere.

 This usually means dividing each component by the Euclidean length of the vector (L2 Norm). In some applications (e.g., histogram features), it can be more practical to use the L1 norm of the feature vector.

Like Min-Max Scaling, the Unit Vector technique produces values of range [0,1]. When dealing with features with hard boundaries, this is quite useful. For example, when dealing with image data, the colors can range from only 0 to 255.

![image-20230704003201861](/images/2023-04-19-Data Preprocessing/image-20230704003201861.png)

If we plot, then it would look as below for L1 and L2 norm, respectively.

![image-20230704003232863](/images/2023-04-19-Data Preprocessing/image-20230704003232863.png)

### StandardScaler

- The Standard Scaler assumes data is normally distributed within each feature and scales them such that the distribution centered around 0, with a standard deviation of 1.
- If data is not normally distributed, this is not the best Scaler to use.
- However, this scaling does not ensure any particular minimum and maximum values for the features.

```python
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler


iris = load_iris()
iris_df = pd.DataFrame(data = iris.data, columns = iris.feature_names)
print('Means of the features:\n{}'.format(iris_df.mean()))
print('Variances of the features:\n{}'.format(iris_df.var()))
```

<pre>
Means of the features:
sepal length (cm)    5.843333
sepal width (cm)     3.057333
petal length (cm)    3.758000
petal width (cm)     1.199333
dtype: float64
Variances of the features:
sepal length (cm)    0.685694
sepal width (cm)     0.189979
petal length (cm)    3.116278
petal width (cm)     0.581006
dtype: float64
</pre>


```python
scaler = StandardScaler()
scaler.fit(iris_df)
iris_scaled = scaler.transform(iris_df) #NumPy ndarray
```


```python
iris_df_scaled = pd.DataFrame(data = iris_scaled, columns = iris.feature_names)
print('Means of the features:\n{}'.format(iris_df_scaled.mean())) # All means are 0.
print('Variances of the features:\n{}'.format(iris_df_scaled.var())) # All variances are 1.
```

<pre>
Means of the features:
sepal length (cm)   -1.690315e-15
sepal width (cm)    -1.842970e-15
petal length (cm)   -1.698641e-15
petal width (cm)    -1.409243e-15
dtype: float64
Variances of the features:
sepal length (cm)    1.006711
sepal width (cm)     1.006711
petal length (cm)    1.006711
petal width (cm)     1.006711
dtype: float64
</pre>

### Max Abs Scaler

- Scale each feature by its maximum absolute value.
- This estimator scales and translates each feature individually such that the maximal absolute value of each feature in the training set is 1.0.
- It does not shift/center the data and thus does not destroy any **sparsity**.
- Cons: On positive-only data, this Scaler behaves similarly to Min Max Scaler and, therefore, also suffers from the presence of significant **outliers**.

![image-20230704002234544](/images/2023-04-19-Data Preprocessing/image-20230704002234544.png)

### Robust Scaler

-  This Scaler is **robust** to outliers.
- If our data contains many **outliers**, scaling using the mean and standard deviation of the data won’t work well.
- The centering and scaling statistics of this Scaler are based on percentiles and are therefore not influenced by a few numbers of huge marginal outliers (The RobustScaler ignore outliers.).
- Note that the outliers themselves are still present in the transformed data. If a **separate outlier clipping** is desirable, a non-linear transformation is required.

![image-20230704002457959](/images/2023-04-19-Data Preprocessing/image-20230704002457959.png)

Let’s now see what happens if we introduce an outlier and see the effect of scaling using Standard Scaler and Robust Scaler (a circle shows outlier).

![image-20230704002541358](/images/2023-04-19-Data Preprocessing/image-20230704002541358.png)

### Quantile Transformer Scaler (**Rank scaler**)

- Transform features using quantiles information.
- This method transforms the features to follow a **uniform or a normal** distribution. 
- Therefore, for a given feature, this transformation tends to spread out the most frequent values.
- It also reduces the impact of (marginal) outliers: this is, therefore, a **robust pre-processing** scheme.
- The cumulative distribution function of a feature is used to project the original values.
- Note that this transform is non-linear and may distort linear correlations between variables measured at the same scale but renders variables measured at different scales more directly comparable.

```python
from sklearn.preprocessing import QuantileTransformer
scaler = QuantileTransformer()
df6 = pd.DataFrame(scaler.fit_transform(df),
                   columns=['WEIGHT','PRICE'],
                   index = ['Orange','Apple','Banana','Grape'])
ax = df.plot.scatter(x='WEIGHT', y='PRICE',color=['red','green','blue','yellow'], 
                     marker = '*',s=80, label='BREFORE SCALING');
df6.plot.scatter(x='WEIGHT', y='PRICE', color=['red','green','blue','yellow'],
                 marker = 'o',s=60,label='AFTER SCALING', ax = ax,figsize=(6,4))
plt.axhline(0, color='red',alpha=0.2)
plt.axvline(0, color='red',alpha=0.2);
```

The above example is just for illustration as Quantile transformer is useful when we have a large dataset with many data points usually more than 1000.

![image-20230704100617082](/images/2023-04-19-Data Preprocessing/image-20230704100617082.png)

### Power Transformer Scaler

- The power transformer is a family of parametric, monotonic transformations that are applied to **make data more Gaussian-like**. 
- This is useful for modeling issues related to the variability of a variable that is unequal across the range (heteroscedasticity) or situations where normality is desired.
- The power transform finds the optimal scaling factor in stabilizing variance and minimizing skewness through maximum likelihood estimation. 
- Currently, Sklearn implementation of PowerTransformer supports the Box-Cox transform and the Yeo-Johnson transform. 
- The optimal parameter for stabilizing variance and minimizing skewness is estimated through maximum likelihood. 
- Box-Cox requires input data to be strictly positive, while Yeo-Johnson supports both positive or negative data.

```python
from sklearn.preprocessing import PowerTransformer
scaler = PowerTransformer(method='yeo-johnson')
df5 = pd.DataFrame(scaler.fit_transform(df),
                   columns=['WEIGHT','PRICE'],
                   index = ['Orange','Apple','Banana','Grape'])
ax = df.plot.scatter(x='WEIGHT', y='PRICE',color=['red','green','blue','yellow'], 
                     marker = '*',s=80, label='BREFORE SCALING');
df5.plot.scatter(x='WEIGHT', y='PRICE', color=['red','green','blue','yellow'],
                 marker = 'o',s=60,label='AFTER SCALING', ax = ax)
plt.axhline(0, color='red',alpha=0.2)
plt.axvline(0, color='red',alpha=0.2);
```

![image-20230704003051614](/images/2023-04-19-Data Preprocessing/image-20230704003051614.png)





# References

1. [Encoding](https://medium.com/towards-data-science/all-about-categorical-variable-encoding-305f3361fd02)
2. [Scaling](https://towardsdatascience.com/all-about-feature-scaling-bcc0ad75cb35)

