---
title:  "SQL: Definitions and Commands"
categories: SQL
tag: [SQL, Data_Cleaning]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

# What are a database and SQL?

## Database

A collection of Information.

- With this kind of basic information in one table, we could use a spreadsheet like Excel to keep track of our data. 
- But databases **allow us not only to add more tables but also** to set up rules and relationships between the tables. 

![image-20230503060238870](/images/2023-05-03-SQL_def_ask_for_data/image-20230503060238870.png)

- Fields and records make up a table, and one or more tables make up a database.
- The layout and definition of how fields, tables, and relationships are set up are called the schema of the database.

## SQL

**SQL stands for Structured Query Language**, which is a language for manipulating and defining data in databases. 

- SQL gives us a way of writing questions a database can understand. 
- SQL is generally `white space independent`, meaning that if you want to add some space or lines between clauses or expressions to make your statement easier to read, you can do so.

### Statement

- Statement: something you write in SQL to get an answer from a database or to make a change.

- **A statement is made up of clauses.** Clauses are the basic components of statements, the smaller building blocks that make up the whole building.

<img src="/images/2023-05-03-SQL_def_ask_for_data/image-20230503060302080.png" alt="image-20230503060302080" style="zoom:50%;" />

These clauses **are constructed out of** various elements, including **keywords**, which are special or reserved words that tell the database to take some action, **field names**, which let us refer to fields or columns of data whose values we want to use, **table names**, which tell the database which table to use, and **predicates**, which we use to specify what information or condition we're looking for. 

<img src="/images/2023-05-03-SQL_def_ask_for_data/image-20230503060322900.png" alt="image-20230503060322900" style="zoom:50%;" />

- **Predicates** include `a value or condition` called an expression. 
- A **clause** can be a statement if you're writing a really basic one. 
- These keywords and operators are customarily written in uppercase, though usually, they don't have to be.
- At the end of a statement, we put a semi-colon.

 SQL statements can also be used to add, modify, or delete data in a database or even to create, modify, and remove tables.

# Ask for Data from a Database

## 1. SELECT

The SELECT keyword tells `the database` we want some information returned to us.

- SELECT first_name FROM people;
- If I wrap that field identifier in single quotes, ‘first_name’, we’d get back **the text string** instead.
- We can change the order of the fields too.
- Wild card operator: the star or asterisk. *

### Mathematics

#### 1) Count ()

```SQL
SELECT LOWER(first_name), UPPER(last_name)
FROM people;
```

If we add a clause to the end using the GROUP BY keyword, we can tell the database to run our SELECT clause **against each individual grouping by the field that we specify.**

#### 2) AVG()

#### 3) SUM()

#### 4) MAX(), MIN()

### Transforming data

#### 1) LOWER(), UPPER()

```SQL
SELECT LOWER(first_name), UPPER(last_name)
FROM people;
```

#### 2) SUBSTR( , , )

In most SQL implementations, substring is the name of the function. But here in SQLite, it's SUBSTR, which is a substring of the word substring. 

: A number representing at which character in the longer string to start counting and the length after which to stop in characters. I want to start with the first character and then proceed with five characters after that to get the first five characters of the last_name field. 

```sql
SELECT first_name, SUBSTR(last_name, 1, 5)
FROM people;
```

#### 3) REPLACE( , , )

```sql
SELECT REPLACE (first_name, "a", "-")
FROM people;
```

<img src="/images/2023-05-03-SQL_def_ask_for_data/image-20230503155843960.png" alt="image-20230503155843960" style="zoom:50%;" />

- This replacement is case-sensitive, so keep an eye out for that.
- In fact, if I scroll down here to row 53, Anne still has a capital A because capital A is a different character than lowercase a. 

### Creating aliases with AS

```sql
SELECT first_name AS firstname, UPPER(last_name) AS surname
FROM people
WHERE firstname = 'Laura';
```

## 2. WHERE

The WHERE keyword lets us add selection criteria to a statement. These clauses need to be in this order to work.

```sql
SELECT first_name, last_name, shirt_or_ hat
FROM people
WHERE shirt or hat='shirt':
```

### Logical operators

- Select all columns with the filter applied
- Logical operator/ = IS/ <>/!= IS NOT
- The database is thinking about those last two conditions as being paired together.

```sql
SELECT first_name, last_name
FROM people
WHERE state_code='CA' AND shirt_or_hat='shirt';
```

### **Like** ‘%’

- Returns results that match part of a string.
- The % character represents the portion of the string to ignore.

```sql
SELECT *
FROM people
WHERE state_code like 'C%';
```

- This tells the database to match the letter C, and then whatever comes after it, we don't care about, regardless of how much information follows the letter C. e="CA", OR state_code="CO", OR state_code="CT", and so on. Or we could say state_code LIKE 'C%' 
- This is `not case-sensitive`. I am using capital letters in my condition, but it’s matching lowercase ones.

## 3. LIMIT n

- Select first n rows for all columns
-  If I wanted to see a specific range of them, like the second set of five, I could use `the offset command` to tell the database `to skip some records` before counting off my five.

```sql
SELECT *
FROM people
WHERE state code like 'C%'
LIMIT 5 OFFSET 5; #The results will be limited to 10 records, skipping the first five records.
```

## 4. ORDER BY

- These are sorted in ascending order.
- The lower the value, the earlier in the list it comes. The larger, the later it comes to the list.

```sql
SELECT first_name, last_name
FROM people
ORDER BY first_name DESC;
```

- I can change the last name sort to descending order while the state is still in **the default ascending order.** And here, I can see that the sort order of the states is still the same, but now the records are reordered in `reverse alphabetical sequence` by the last name within the state. 

```sql
SELECT state_code, frst_name, last_name
FROM people
ORDER BY state_code, last_name DESC;
```

## 5. Functions: Finding information about the data

It tells us how long the information in the given field is in characters rather than the value of the field itself.

\- Janice is six characters long.

```sql
SELECT first_name, length(first_name)
FROM people
```

Let's take a look at pulling out only the unique values or the values that are distinct from one another with the DISTINCT function. 

```sql
SELECT DISTINCT(frst_name)
FROM people
ORDER by frst_name;
```

