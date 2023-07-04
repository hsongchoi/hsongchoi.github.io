---
title:  "Feature Selection"
categories: ML
tag: [Scikit-learn, Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: true

---

![image-20230704123842418](../images/2023-07-04-Feature_selection/image-20230704123842418.png)

Selecting which features to use is a crucial step in any machine learning project and a recurrent task in the day-to-day of a Data Scientist.

- Two of the biggest problems in Machine Learning are 1. **overfitting** (fitting aspects of data that are not generalizable outside the dataset) and 2. the **curse of dimensionality** (the unintuitive and sparse properties of data in high dimensions).
- Feature selection helps to avoid both of these problems by reducing the number of features in the model and trying to optimize the model performance. 
- In doing so, feature selection also provides an extra benefit: **Model** **interpretation**. With fewer features, the output model becomes simpler and easier to interpret, and it becomes more likely for a human to trust future predictions made by the model.

![image-20230704123501123](../images/2023-07-04-Feature_selection/image-20230704123501123.png)

# Unsupervised methods

