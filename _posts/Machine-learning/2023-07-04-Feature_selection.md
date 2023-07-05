---
title:  "Feature Selection"
categories: ML
tag: [Scikit-learn, Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: true

---

![image-20230704123842418](/images/2023-07-04-Feature_selection/image-20230704123842418.png)

Selecting which features to use is a crucial step in any machine learning project and a recurrent task in the day-to-day of a Data Scientist.

- Two of the biggest problems in Machine Learning are 
  - **Overfitting** (fitting aspects of data that are not generalizable outside the dataset)
  - the **Curse of dimensionality** (the unintuitive and sparse properties of data in high dimensions).
- Feature selection helps to avoid both of these problems by reducing the number of features in the model and trying to optimize the model performance. 
- In doing so, feature selection also provides an extra benefit: **Model** **interpretation**. With fewer features, the output model becomes simpler and easier to interpret, and it becomes more likely for a human to trust future predictions made by the model.

![image-20230704123501123](/images/2023-07-04-Feature_selection/image-20230704123501123.png)

# Unsupervised methods

- One simple method to reduce the number of features consists of applying a **Dimensionality Reduction technique** to the data.
- Dimensionality reduction does not actually select a subset of features but instead produces a new set of features in a lower dimension space. 
- In practice, we perform dimensionality reduction (e.g. PCA) over a subset of features and check how the labels are distributed in the reduced space. If they appear to be **separate**, this is a clear sign that **high classification performance** is expected when using this set of features.

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
h = .01
x_min, x_max = -4,4
y_min, y_max = -1.5,1.5
# loading dataset
data = load_iris()
X, y = data.data, data.target
# selecting first 2 components of PCA
X_pca = PCA().fit_transform(X)
X_selected = X_pca[:,:2]
# training classifier and evaluating on the whole plane
clf = SVC(kernel='linear')
clf.fit(X_selected,y)
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
# Plotting
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.figure(figsize=(10,5))
plt.pcolormesh(xx, yy, Z, alpha=.6,cmap=cmap_light)
plt.title('PCA - Iris dataset')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.scatter(X_pca[:,0],X_pca[:,1],c=data.target,cmap=cmap_bold)
plt.show()
```

![image-20230704173529020](/images/2023-07-04-Feature_selection/image-20230704173529020.png)

## PCA

- From a simplified perspective, PCA transforms data linearly into new properties that are not correlated with each other. 
- What is the difference between SVD and PCA?
  -  SVD gives you the whole nine-yard of diagonalizing a matrix into [special matrices](https://medium.com/@jonathan_hui/machine-learning-linear-algebra-special-matrices-c750cd742dfe) that are easy to manipulate and to analyze.
  - Obviously, we can use SVD to find PCA by truncating the less important basis vectors in the original SVD matrix.

A matrix can be diagonalized if *A* is a $n \times n$ square matrix and *A* has n linearly independent eigenvectors. 

![image-20230704182540847](/images/2023-07-04-Feature_selection/image-20230704182540847.png)

