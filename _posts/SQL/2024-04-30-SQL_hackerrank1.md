---
title: "SQL: Hacker Rank Questions 1"
categories: SQL
tag: [SQL]
author_profile: false
typora-root-url: ../
search: true
use_math: true

---

# Q1. Weather Observation Station 3

![image-20240430141043603](/images/2024-04-30-SQL_hackerrank1/image-20240430141043603.png)

```sql
SELECT DISTINCT(CITY)
FROM STATION
WHERE ID % 2 = 0
```

# Q2. Weather Observation Station 4

![image-20240430142339948](/images/2024-04-30-SQL_hackerrank1/image-20240430142339948.png)

```sql
SELECT (COUNT(CITY) - COUNT(DISTINCT(CITY) )) AS DIFFERENCE
FROM STATION
```

# Q3. Weather Observation Station 5

![image-20240430143227827](/images/2024-04-30-SQL_hackerrank1/image-20240430143227827.png)

```mysql
SELECT CITY, LENGTH(CITY) AS LENGTH
FROM STATION
ORDER BY LENGTH, CITY
LIMIT 1;

SELECT CITY, LENGTH(CITY) AS LENGTH
FROM STATION
ORDER BY LENGTH DESC, CITY
LIMIT 1;
```

# Q4. Weather Observation Station 6

![image-20240430152819628](/images/2024-04-30-SQL_hackerrank1/image-20240430152819628.png)

```sql
SELECT DISTINCT CITY 
FROM STATION 
WHERE LEFT(CITY, 1) IN ("A", "E", "I", "O", "U")
-- WHERE RIGHT(CITY, 1) IN ("A", "E", "I", "O", "U")

```

# Q5. Weather Observation Station 8

![image-20240430153742603](/images/2024-04-30-SQL_hackerrank1/image-20240430153742603.png)

```sql
SELECT DISTINCT CITY 
FROM STATION 
WHERE LEFT(CITY, 1) IN ("A", "E", "I", "O", "U")
    AND
    RIGHT(CITY, 1) IN ("A", "E", "I", "O", "U")
```

# Q6. Weather Observation Station 9

![image-20240430154126484](/images/2024-04-30-SQL_hackerrank1/image-20240430154126484.png)

```sql
SELECT DISTINCT CITY 
FROM STATION 
WHERE LEFT(CITY, 1) NOT IN ("A", "E", "I", "O", "U")
```

# Q7. Weather Observation Station 11

![image-20240430154339485](/images/2024-04-30-SQL_hackerrank1/image-20240430154339485.png)

```sql
SELECT DISTINCT CITY 
FROM STATION 
WHERE LEFT(CITY, 1) NOT IN ("A", "E", "I", "O", "U")
    OR 
    RIGHT(CITY, 1) NOT IN ("A", "E", "I", "O", "U")
    
-- !AND = OR
```

# Q8. Higher Than 75 Marks

![image-20240509141852169](/images/2024-04-30-SQL_hackerrank1/image-20240509141852169.png)

```sql
SELECT NAME
FROM STUDENTS
WHERE MARKS >75
ORDER BY RIGHT(NAME, 3), ID
```

# Q9. Type of Triangle

![image-20240509144016704](/images/2024-04-30-SQL_hackerrank1/image-20240509144016704.png)

```sql
SELECT CASE
    WHEN (a=b and b=c) THEN 'Equilateral' 
    WHEN (a+b<=c or b+c<=a or c+a<=b) THEN 'Not A Triangle' 
    WHEN (a!=b and b!=c and c!=a) THEN 'Scalene' 
ELSE 'Isosceles' END
FROM triangles
```

```sql
SELECT CONCAT( Name,'(', LEFT(occupation,1),')' ) FROM OCCUPATIONS ORDER BY Name ASC;

SELECT CONCAT("There are a total of ",COUNT(occupation)," ",LOWER(occupation),"s.") FROM OCCUPATIONS GROUP BY occupation ORDER BY COUNT(occupation) ASC, LOWER(occupation) ASC;
```

