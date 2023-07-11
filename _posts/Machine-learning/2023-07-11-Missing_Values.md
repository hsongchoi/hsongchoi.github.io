---
title:  "Handling Missing Values"
categories: ML
tag: [Scikit-learn, Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---



A perfect data set is usually a big win for any data scientist or machine learning engineer. Unfortunately, more often than not, datasets will have missing data.

![image-20230711171144903](/images/2023-07-11-Missing_Values/image-20230711171144903.png)

# Causes of Missing Data

- Data is not being intentionally filled, especially if it is an optional field.
- Data being corrupted.
- Human error.
- If it was a survey, participants might quit the survey halfway.
- If data is being automatically by computer applications, then a malfunction could cause missing data. Eg. a sensor recording logs malfunctioning.
- Fraudulent behavior of intentionally deleting data.

# Types of Missing Data

1. **Missing Completely at Random (MCAR)**
   1. This effectively implies that the causes of the missing data are unrelated to the data.
   2. It is safe to ignore many of the complexities that arise because of the missing data, apart from the obvious loss of information.
   3. **Example:** Estimate the gross annual income of a household within a certain population, which you obtain via questionnaires. In the case of MCAR, the missingness is completely random, as if some questionnaires were lost **by mistake**.
2. **Missing at Random (MAR)**
   1. If the probability of being missing is the same only **within groups** defined by the *observed* data, then the data are missing at random (MAR).
   2. For instance, suppose you also collected data on the profession of each subject in the questionnaire and deduce that managers, VIPs, etc are more likely not the share their income. Then, within subgroups of the profession, missingness is random.

3. **Not Missing at Random (NMAR)**
   1. If neither MCAR nor MAR holds, then we speak of missing not at random (MNAR).
   2. **Example:** In the case of MNAR when the reason for missingness depends on the missing values themselves. For instance, suppose people don’t want to share their income as it is less, and they are ashamed of it.

# Ways to Handle Missing Values

Pandas write the value `NaN`(Not a Number) when it finds a missing value.

## Checking Missing Values

![image-20230711172100634](/images/2023-07-11-Missing_Values/image-20230711172100634.png)

## The easy way:

1. **Deleting rows that have missing values.**
   1. This technique can lead to loss of information and **hence should not be used when the count of rows with missing values is high.**
   2. ![image-20230711172419492](/images/2023-07-11-Missing_Values/image-20230711172419492.png)

2. **Deleting columns with missing values**.
   1. If we have columns that have extremely high missing values in them (say 80% of the data in the columns is missing), then we can delete these columns.
   2. ![image-20230711172506927](/images/2023-07-11-Missing_Values/image-20230711172506927.png)
   3. With the above output, you can now decide to set a threshold for the percentage of the missing values you want to delete.
   4. ![image-20230711172552983](/images/2023-07-11-Missing_Values/image-20230711172552983.png)
   5. From the output Building Area and YearBuilt are the columns we will delete using the drop function.
      1. ![image-20230711172629422](/images/2023-07-11-Missing_Values/image-20230711172629422.png)
   6. *It is an option only if the number of missing values is 2% of the whole dataset or less.* 
   7. **Do not use this as your first approach.**
3. **Leave it to the algorithm**
   1. Some algorithms can factor in the missing values and learn the best imputation values for the missing data based on the training loss reduction (ie. XGBoost). 
   2. Some others have the option to just ignore them (ie. LightGBM — *use_missing=false*).
   3. However, other algorithms throw an error about the missing values (ie. Scikit learn — LinearRegression).
   4. *Is an option only if the missing values are about 5% or less. Works with MCAR.*

## The professional way:

The drawback of dropping missing values is that you loose the entire row just for the a few missing values. That is a lot of valuable data. 

- Try filling in the missing values with a well-calculated estimate.
- Professionals use two main methods of calculating missing values. They are **imputation** and **interpolation.**

### Imputation

#### Mean/Median Imputation

Numerical Data: **Replacing missing values with mean, mode, or median** 

1. Note that this method can add variance and bias error.
2. ![image-20230711174115646](/images/2023-07-11-Missing_Values/image-20230711174115646.png)
3. Your business rules could also have specified values to fill your missing values with. You can just specify the value directly as well, for instance, I am replacing BuildingSize with size 100 by default for all missing values.
   1. ![image-20230711174213633](/images/2023-07-11-Missing_Values/image-20230711174213633.png)

**Advantages:**

- Quick and easy
- Ideal for small numerical datasets

**Disadvantages:**

- Doesn’t factor in the correlations between features. It only works on the column level.
- It will give poor results on encoded categorical features (do NOT use it on categorical features).
- Not very accurate.
- Doesn’t account for the uncertainty in the imputations.

#### Most Frequent (Values) Imputation

Categorical Data: **Replacing missing values with the most frequent**

1. Imputing using mean, mode, and median works best with numerical values. For categorical data, we can impute using the most frequent or constant value.

2. Let’s use the sklearn impute package to replace categorical data with the most frequent value by specifying **strategy=’most_frequent’** ![image-20230711174357774](/images/2023-07-11-Missing_Values/image-20230711174357774.png) The code above will replace Regionname with the most frequent region.

3. Let’s use the sklearn impute package to replace categorical data with a constant value by specifying **strategy=’constant’.** You also need to include which value is going to be filler by specifying the fill_value. In our case, we are going to fill missing values in column ‘CouncilArea’ with a value called ‘other’. This technique can also be used when there are set business rules for the use case in context. ![image-20230711174543956](/images/2023-07-11-Missing_Values/image-20230711174543956.png)

4. You can now see below we have a new category called other added to the ‘CouncilArea’ column.

   ![image-20230711174700702](/images/2023-07-11-Missing_Values/image-20230711174700702.png)

**Advantages:**

- Works well with categorical features.

**Disadvantages:**

- It also doesn’t factor in the correlations between features.
- It can introduce bias in the data.

#### Zeros Imputation

- It replaces the missing values with either zero or any constant value you specify.

- Perfect for when the null value does not add value to your analysis but requires an integer in order to produce results.

#### Regression Imputation

Instead of just taking the mean, you’re taking the predicted value based on other variables. This preserves relationships among variables involved in the imputation model but not variability around predicted values.

5. **Predicting missing values using algorithms**
   1. If you want to replace a **categorical value**, use **the classification algorithm**.
   2. If predicting a **continuous number**, we use a **regression algorithm**. This method is also good because it generates unbiased estimates.

#### Stochastic Regression Imputation

The predicted value from a regression plus a random residual value. This has all the advantages of regression imputation but adds in the advantages of the random component.

#### Imputaiton using k-NN

- The *k* nearest neighbors is an algorithm that is used for simple classification.
- The algorithm uses ‘**feature similarity**’ to predict the values of any new data points. 
- This can be very useful in making predictions about the missing values by finding the *k’s* closest neighbors to the observation with missing data and then imputing them based on the non-missing values in the neighborhood.
- The process is as follows: Basic mean impute -> KDTree -> compute nearest neighbors (NN) -> Weighted average.

**Pros:**

- Can be much more accurate than the mean, median or most frequent imputation methods (It depends on the dataset).

**Cons:**

- Computationally expensive. KNN works by storing the whole training dataset in memory.
- K-NN is quite sensitive to outliers in the data.

#### Imputation Using Multivariate Imputation by Chained Equation (MICE)

This type of imputation works by filling in the missing data multiple times. Multiple Imputations (MIs) are much better than a single imputation as it measures the uncertainty of the missing values in a better way.

The chained equations approach is also very flexible and can handle different variables of different data types (i.e., continuous or binary) as well as complexities such as bounds or survey skip patterns.

```R
install.package("mice") #install mice
install.package("lattice")#install lattice
library("mice") #load mice
library("lattice") #load lattice

micedata <- mice(mtcars[, !names(mtcars) %in% "cyl"], method="rf")  # perform mice imputation, based on random forests.
miceOutput <- complete(micedata)  # generate the completed data.

#Check for NAs
anyNA(miceOutput)
```

When to use Imputation or Interpolation:

**Imputation**: When you require the mean or median of a set of values.

Eg, If there are missing values in a dataset of marks obtained by students, you can impute missing values with the mean of the other student's marks.

**Interpolation**: When you have data points with a linear relationship.

Eg: The growth chart of a healthy child. The child will grow taller with each passing year until they are 16. Any missing values along this growth chart will be part of a straight line, so the missing values can be interpolated.

- [Reference1](https://medium.com/mlearning-ai/handling-missing-values-data-science-7b8e302264ee)
- [Reference2](https://medium.com/mlearning-ai/handling-missing-values-data-science-7b8e302264ee)



