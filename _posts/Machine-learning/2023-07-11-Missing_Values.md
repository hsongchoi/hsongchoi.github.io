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

# Handling Missing Values

How you handle missing values will depend on your problem statement and your data. There are 3 main ways to handle missing values.

1. Removing the columns having missing values — If you have a column with more than 80% missing values, then it is better to drop the column.
2. Removing the rows having missing values — If you have a small percentage of rows with missing values, then you can just drop those rows. If, say, you have 1 Million records, and out of that, just 20 rows have missing values, then you can drop those rows since they won’t take away a lot of information from your data.
3. **Imputing the missing values** — This is usually the preferred way of handling the missing values, especially where the missing values are **not more than 80%**. 
   - Imputing means filling the missing values with a valid value, for instance, replacing missing values with **the average, mode, or median**.

## Handling Missing Values in Pandas

Pandas write the value `NaN`(Not a Number) when it finds a missing value.

### Checking Missing Values

![image-20230711172100634](/images/2023-07-11-Missing_Values/image-20230711172100634.png)

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

3. Numerical Data: **Replacing missing values with mean, mode, or median(Numerical data)** 

   1. Note that this method can add variance and bias error.
   2. ![image-20230711174115646](/images/2023-07-11-Missing_Values/image-20230711174115646.png)
   3. Your business rules could also have specified values to fill your missing values with. You can just specify the value directly as well, for instance, I am replacing BuildingSize with size 100 by default for all missing values.
      1. ![image-20230711174213633](/images/2023-07-11-Missing_Values/image-20230711174213633.png)

4. Categorical Data: **Replacing missing values with most frequent or constant value**

   1. Imputing using mean, mode, and median works best with numerical values. For categorical data, we can impute using the most frequent or constant value.

   2. Let’s use the sklearn impute package to replace categorical data with the most frequent value by specifying **strategy=’most_frequent’** ![image-20230711174357774](/images/2023-07-11-Missing_Values/image-20230711174357774.png) The code above will replace Regionname with the most frequent region.

   3. Let’s use the sklearn impute package to replace categorical data with a constant value by specifying **strategy=’constant’.** You also need to include which value is going to be filler by specifying the fill_value. In our case, we are going to fill missing values in column ‘CouncilArea’ with a value called ‘other’. This technique can also be used when there are set business rules for the use case in context. ![image-20230711174543956](/images/2023-07-11-Missing_Values/image-20230711174543956.png)

   4. You can now see below we have a new category called other added to the ‘CouncilArea’ column.

      ![image-20230711174700702](/images/2023-07-11-Missing_Values/image-20230711174700702.png)

5. **Predicting missing values using algorithms**
   1. If you want to replace a **categorical value**, use **the classification algorithm**.
   2. If predicting a **continuous number**, we use a **regression algorithm**. This method is also good because it generates unbiased estimates.





