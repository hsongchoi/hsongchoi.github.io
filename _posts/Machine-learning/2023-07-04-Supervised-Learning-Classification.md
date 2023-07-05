---
title:  "Supervised Learning-Classification"
categories: ML
tag: [Scikit-learn, Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

![image-20230705092329657](/images/2023-07-04-Supervised-Learning-Classification/image-20230705092329657.png)

Classification is one of the most important aspects of **supervised learning**.

# Logistic Regression



# Fisher’s Linear Discriminant

## Background

Consider the binary classification (**K=2**)—blue and red points in **R²**. Here, **D** represents the original input dimensions while **D’** is the projected space dimensions. Throughout this article, consider **D’** less than **D**. In the case of projecting to one dimension (the number line), i.e., **D’=1**, we can pick a threshold **t** to separate the classes in the new space. 

- We want to reduce the original data dimensions from **D=2** to **D’=1**. 
- First, compute the mean vectors **m1** and **m2** for the two classes.

![image-20230705093724265](/images/2023-07-04-Supervised-Learning-Classification/image-20230705093724265.png)

- Note that **N1** and **N2** denote the number of points in classes C1 and C2, respectively.
- Now, consider using the class means as a measure of separation. In other words, we want to project the data onto the vector **W** joining the 2 class means.
- In this scenario, note that the two classes are clearly separable (by a line) in their original space. 
- However, after re-projection, the data exhibit some sort of class overlapping — shown by the yellow ellipse on the plot and the histogram below.

![image-20230705094512229](/images/2023-07-04-Supervised-Learning-Classification/image-20230705094512229.png)

That is where Fisher’s Linear Discriminant comes into play.

- The idea proposed by Fisher is to **maximize a function** that will **give a large separation** between the projected class means while also giving **a small variance within each class**, thereby **minimizing the class overlap**.
- In other words, FLD selects a projection that maximizes class separation. To do that, it **maximizes the ratio between the between-class variance to the within-class variance.**
- A large variance among the dataset classes./ A small variance within each of the dataset classes.

![image-20230705094750697](/images/2023-07-04-Supervised-Learning-Classification/image-20230705094750697.png)

To find the projection with the following properties, FLD learns a weight vector **W** with the following criterion.

-  Construct the lower dimensional space, which maximizes the between-class variance and minimizes the within-class variance.
- Let **W**  be the lower dimensional space projection, which is called Fisher’s criterion.

![image-20230705095330566](/images/2023-07-04-Supervised-Learning-Classification/image-20230705095330566.png)

where

![image-20230705095402302](/images/2023-07-04-Supervised-Learning-Classification/image-20230705095402302.png)

![image-20230705095457443](/images/2023-07-04-Supervised-Learning-Classification/image-20230705095457443.png)

![image-20230705095526299](/images/2023-07-04-Supervised-Learning-Classification/image-20230705095526299.png)

## FDA for Multiple Classes

We can generalize FLD for the case of more than **K>2** classes. Here, we need generalization forms for the **within-class** and **between-class** covariance matrices.

![image-20230705095659634](/images/2023-07-04-Supervised-Learning-Classification/image-20230705095659634.png)

- To find the weight vector **W**, we take the **D’** eigenvectors that correspond to their largest eigenvalues (equation 8).
- In other words, if we want to reduce our data dimensions from **D=784** to **D’=2**, the transformation vector **W** is composed of the 2 eigenvectors that correspond to the **D’=2** largest eigenvalues. This gives a final shape of **W = (N,D’)**, where **N** is the number of input records and **D’** the reduced feature space dimensions.

## Building a linear discriminant

Up until this point, we used Fisher’s Linear discriminant only as a method for dimensionality reduction. To really create a discriminant, we can model a **multivariate Gaussian distribution** over a D-dimensional input vector **x** for each class **K** as:

![image-20230705095943887](/images/2023-07-04-Supervised-Learning-Classification/image-20230705095943887.png)

Here ***μ\*** (the mean) is a D-dimensional vector. **Σ** (sigma) is a **DxD** matrix — the covariance matrix. And |**Σ**| is the determinant of the covariance. The determinant is a measure of how much the covariance matrix **Σ** stretches or shrinks space.

- For multiclass data, we can (1) model a class conditional distribution using a Gaussian. (2) Find the prior class probabilities *P(Ck),* and (3) use **Bayes** to find the posterior class probabilities *p(Ck|x)*.

- The parameters of the Gaussian distribution: ***μ\*** and **Σ,** are computed for each class **k=1,2,3,…, K** using the projected input data**.** We can infer the priors *P(Ck)* class probabilities using the fractions of the training set data points in each of the classes (line 11).

  Once we have the Gaussian parameters and priors, we can compute class-conditional densities *P(****x\****|Ck)* for each class **k=1,2,3,…, K** individually. To do it, we first project the D-dimensional input vector **x** to a new **D’** space. Keep in mind that **D’ < D**. Then, we evaluate equation 9 for each projected point. Finally, we can get the posterior class probabilities *P(Ck|****x\****)* for each class **k=1,2,3,…, K** using equation 10.

![image-20230705100355022](/images/2023-07-04-Supervised-Learning-Classification/image-20230705100355022.png)

## What are the steps in Linear Discriminant Analysis (LDA)?

1. Calculate the mean for each class and the overall mean
2. Calculate the within-class covariance matrix
3. Calculate the between-class covariance matrix
4. Calculate the eigenvalues and eigenvectors of the covariance matrices
5. Determine the linear discrimination functions
6. Construct a decision surface

## Conclusion

- LDA supports both binary and multi-class classification. It can be used for both binary and multiclass problems as well as to effectively shrink the number of features in the model.
- **LDA inherits the problems of GLM.** **It assumes the Gaussian distribution of the input variables.** Always consider reviewing the univariate distributions of each attribute and using transforms to make them more Gaussian-looking.
- Like GLM, **outliers will affect the mean.** Also, skew and kurtosis need to be checked for as they affect standard deviation.
- Features assume that **each input variable has the same variance.**
- It will help to standardize the features and also limit the standard deviation between 0 to 1.

# SVM

