---
title: "SQL: Strata Scratch test"
categories: SQL
tag: [SQL, Data_Cleaning]
author_profile: false
typora-root-url: ../
search: true
use_math: true

---

This is for the interview of SQL.

- [Reference 1](https://www.stratascratch.com/blog/database-interview-questions/)

# SELECT Clause

## Easy questions

### [584. Find Customer Referee](https://leetcode.com/problems/find-customer-referee/)

Find the names of the customers that are not referred by the customer with `id=2`.

Return the result table in any order.

```
Input: 
Customer table:
+----+------+------------+
| id | name | referee_id |
+----+------+------------+
| 1  | Will | null       |
| 2  | Jane | null       |
| 3  | Alex | 2          |
| 4  | Bill | null       |
| 5  | Zack | 1          |
| 6  | Mark | 2          |
+----+------+------------+
Output: 
+------+
| name |
+------+
| Will |
| Jane |
| Bill |
| Zack |
+------+
```



```sql
SELECT name
FROM Customer
WHERE referee_id != 2 # <> 2 
	OR referee_id IS NULL
```

### 1148. [Article Views I](https://leetcode.com/problems/article-views-i/)

Write a solution to find all the authors who viewed at least one of their own articles. Return the result table sorted by `id` in ascending order.

```
Input: 
Views table:
+------------+-----------+-----------+------------+
| article_id | author_id | viewer_id | view_date  |
+------------+-----------+-----------+------------+
| 1          | 3         | 5         | 2019-08-01 |
| 1          | 3         | 6         | 2019-08-02 |
| 2          | 7         | 7         | 2019-08-01 |
| 2          | 7         | 6         | 2019-08-02 |
| 4          | 7         | 1         | 2019-07-22 |
| 3          | 4         | 4         | 2019-07-21 |
| 3          | 4         | 4         | 2019-07-21 |
+------------+-----------+-----------+------------+
Output: 
+------+
| id   |
+------+
| 4    |
| 7    |
+------+
```



```sql
SELECT DISTINCT author_id AS id
FROM Views 
WHERE author_id=viewer_id
ORDER BY author_id
```

### [1731. The Number of Employees Which Report to Each Employee](https://leetcode.com/problems/the-number-of-employees-which-report-to-each-employee/)

For this problem, we will consider a manager an employee who has at least 1 other employee reporting to them.

Write a solution to report the ids and the names of all managers, the number of employees who report directly to them, and the average age of the reports rounded to the nearest integer.

Return the result table ordered by `employee_id`.

```
Input: 
Employees table:
+-------------+---------+------------+-----+
| employee_id | name    | reports_to | age |
+-------------+---------+------------+-----+
| 9           | Hercy   | null       | 43  |
| 6           | Alice   | 9          | 41  |
| 4           | Bob     | 9          | 36  |
| 2           | Winston | null       | 37  |
+-------------+---------+------------+-----+
Output: 
+-------------+-------+---------------+-------------+
| employee_id | name  | reports_count | average_age |
+-------------+-------+---------------+-------------+
| 9           | Hercy | 2             | 39          |
+-------------+-------+---------------+-------------+
Explanation: Hercy has 2 people report directly to him, Alice and Bob. Their average age is (41+36)/2 = 38.5, which is 39 after rounding it to the nearest integer.
```

```sql
SELECT 
  e1.employee_id
  , e1.name
  , COUNT(*) AS reports_count
  , ROUND(AVG(e2.age), 0) AS average_age
FROM Employees e1, Employees e2
WHERE e1.employee_id=e2.reports_to
GROUP BY 1, 2
ORDER BY 1
```

### [180. Consecutive Numbers](https://leetcode.com/problems/consecutive-numbers/)

Find all numbers that appear at least three times consecutively.

Return the result table in any order.

```
Input: 
Logs table:
+----+-----+
| id | num |
+----+-----+
| 1  | 1   |
| 2  | 1   |
| 3  | 1   |
| 4  | 2   |
| 5  | 1   |
| 6  | 2   |
| 7  | 2   |
+----+-----+
Output: 
+-----------------+
| ConsecutiveNums |
+-----------------+
| 1               |
+-----------------+
Explanation: 1 is the only number that appears consecutively for at least three times.
```

```sql
WITH cte AS (
  SELECT
    num
    , LEAD(num, 1) OVER() AS lead1
    , LEAD(num, 2) OVER() AS lead2
  FROM logs
)
SELECT DISTINCT num AS ConsecutiveNums
FROM cte
WHERE num=lead1 AND lead1=lead2
```

Create two columns for leading values, one succeeding by one row and the other succeeding by two rows.

If the current number equals the next two, then the number is considered to appear consecutively at least three times.

- The LEAD() function is a **window function** that allows you to access the **next row’s** value **without using** JOIN. It is commonly used in **time-series analysis** and **comparisons** within ordered data.

- LEAD(column_name, offset, default_value) OVER (PARTITION BY partition_column ORDER BY order_column)

- **Parameters:**

  ​	•	column_name → The column from which to retrieve the value.

  ​	•	offset (Optional) → The number of rows ahead (default = 1).

  ​	•	default_value (Optional) → A value to return if no next row exists.

  **Difference Between LEAD() and LAG()**

  - LEAD(): Gets **next** row’s value
  - LAG(): Gets **previous** row’s value

# JOINs

## Easy-level questions

### [1378. Replace Employee ID With The Unique Identifier](https://leetcode.com/problems/replace-employee-id-with-the-unique-identifier/)

Write a solution to show the unique ID of each user. If a user does not have a unique ID, replace just `null`.

Return the result table in any order.

```
Input: 
Employees table:
+----+----------+
| id | name     |
+----+----------+
| 1  | Alice    |
| 7  | Bob      |
| 11 | Meir     |
| 90 | Winston  |
| 3  | Jonathan |
+----+----------+
EmployeeUNI table:
+----+-----------+
| id | unique_id |
+----+-----------+
| 3  | 1         |
| 11 | 2         |
| 90 | 3         |
+----+-----------+
Output: 
+-----------+----------+
| unique_id | name     |
+-----------+----------+
| null      | Alice    |
| null      | Bob      |
| 2         | Meir     |
| 3         | Winston  |
| 1         | Jonathan |
+-----------+----------+
Explanation: 
Alice and Bob do not have a unique ID, We will show null instead.
The unique ID of Meir is 2.
The unique ID of Winston is 3.
The unique ID of Jonathan is 1.
```



```sql
SELECT eu.unique_id, e.name
FROM Employees e
LEFT JOIN EmployeeUNI eu ON e.id = eu.id

```

### [1581. Customer Who Visited but Did Not Make Any Transactions](https://leetcode.com/problems/customer-who-visited-but-did-not-make-any-transactions/)

Write a solution to find the IDs of the users who visited without making any transactions and the number of times they made these types of visits.

Return the result table sorted in any order.

```
Input: 
Visits
+----------+-------------+
| visit_id | customer_id |
+----------+-------------+
| 1        | 23          |
| 2        | 9           |
| 4        | 30          |
| 5        | 54          |
| 6        | 96          |
| 7        | 54          |
| 8        | 54          |
+----------+-------------+
Transactions
+----------------+----------+--------+
| transaction_id | visit_id | amount |
+----------------+----------+--------+
| 2              | 5        | 310    |
| 3              | 5        | 300    |
| 9              | 5        | 200    |
| 12             | 1        | 910    |
| 13             | 2        | 970    |
+----------------+----------+--------+
Output: 
+-------------+----------------+
| customer_id | count_no_trans |
+-------------+----------------+
| 54          | 2              |
| 30          | 1              |
| 96          | 1              |
+-------------+----------------+
Explanation: 
Customer with id = 23 visited the mall once and made one transaction during the visit with id = 12.
Customer with id = 9 visited the mall once and made one transaction during the visit with id = 13.
Customer with id = 30 visited the mall once and did not make any transactions.
Customer with id = 54 visited the mall three times. During 2 visits they did not make any transactions, and during one visit they made 3 transactions.
Customer with id = 96 visited the mall once and did not make any transactions.
As we can see, users with IDs 30 and 96 visited the mall one time without making any transactions. Also, user 54 visited the mall twice and did not make any transactions.
 
```

```sql
SELECT customer_id, count(v.visit_id) as count_no_trans
FROM Visits v
LEFT JOIN Transactions t 
ON v.visit_id = t.visit_id
WHERE t.transaction_id is NULL
GROUP BY 1
```

### [197. Rising Temperature](https://leetcode.com/problems/rising-temperature/)

Write a solution to find all date IDs with higher temperatures compared to its previous dates (yesterday).

Return the result table in any order.

```
Input: 
Weather table:
+----+------------+-------------+
| id | recordDate | temperature |
+----+------------+-------------+
| 1  | 2015-01-01 | 10          |
| 2  | 2015-01-02 | 25          |
| 3  | 2015-01-03 | 20          |
| 4  | 2015-01-04 | 30          |
+----+------------+-------------+
Output: 
+----+
| id |
+----+
| 2  |
| 4  |
+----+
Explanation: 
In 2015-01-02, the temperature was higher than the previous day (10 -> 25).
In 2015-01-04, the temperature was higher than the previous day (20 -> 30).
```

```sql
SELECT w1.id
FROM Weather w1, Weather w2
WHERE w1.temperature > w2.temperature
  AND DATEDIFF(w1.recordDate, w2.recordDate) = 
```

```sql
SELECT y.id
FROM Weather x
LEFT JOIN Weather y ON x.recordDate + INTERVAL 1 DAY = y.recordDate 
# LEFT JOIN Weather y ON x.recordDate + 1 = y.recordDate 
WHERE x.temperature < y.temperature
```

### [1661. Average Time of Process per Machine](https://leetcode.com/problems/average-time-of-process-per-machine/)

There is a factory website that has several machines each running the same number of processes. Write a solution to find the average time each machine takes to complete a process.

The time to complete a process is the `end` timestamp minus the `start` timestamp. The average time is calculated by the total time to complete every process on the machine divided by the number of processes that were run.

The resulting table should have the `machine_id` along with the average time as `processing_time`, which should be rounded to 3 decimal places.

Return the result table in any order.

```
Input: 
Activity table:
+------------+------------+---------------+-----------+
| machine_id | process_id | activity_type | timestamp |
+------------+------------+---------------+-----------+
| 0          | 0          | start         | 0.712     |
| 0          | 0          | end           | 1.520     |
| 0          | 1          | start         | 3.140     |
| 0          | 1          | end           | 4.120     |
| 1          | 0          | start         | 0.550     |
| 1          | 0          | end           | 1.550     |
| 1          | 1          | start         | 0.430     |
| 1          | 1          | end           | 1.420     |
| 2          | 0          | start         | 4.100     |
| 2          | 0          | end           | 4.512     |
| 2          | 1          | start         | 2.500     |
| 2          | 1          | end           | 5.000     |
+------------+------------+---------------+-----------+
Output: 
+------------+-----------------+
| machine_id | processing_time |
+------------+-----------------+
| 0          | 0.894           |
| 1          | 0.995           |
| 2          | 1.456           |
+------------+-----------------+
Explanation: 
There are 3 machines running 2 processes each.
Machine 0's average time is ((1.520 - 0.712) + (4.120 - 3.140)) / 2 = 0.894
Machine 1's average time is ((1.550 - 0.550) + (1.420 - 0.430)) / 2 = 0.995
Machine 2's average time is ((4.512 - 4.100) + (5.000 - 2.500)) / 2 = 1.456
```

```sql
select
machine_id
, round(sum(if(activity_type = 'start', -1, 1) *timestamp)/count(distinct process_id),3) as processing_time
from activity
group by machine_id
```

```sql
SELECT 
  a1.machine_id
  , ROUND(AVG(a2.timestamp - a1.timestamp), 3) AS processing_time
FROM Activity a1
INNER JOIN Activity a2
ON a1.machine_id=a2.machine_id
  AND a1.process_id=a2.process_id
  AND a1.activity_type='start'
  AND a2.activity_type='end'
GROUP BY a1.machine_id
```

### [577. Employee Bonus](https://leetcode.com/problems/employee-bonus/)

Write a solution to report the name and bonus amount of each employee with a bonus of less than 1,000.

Return the result table in any order.

```
Input: 
Employee table:
+-------+--------+------------+--------+
| empId | name   | supervisor | salary |
+-------+--------+------------+--------+
| 3     | Brad   | null       | 4000   |
| 1     | John   | 3          | 1000   |
| 2     | Dan    | 3          | 2000   |
| 4     | Thomas | 3          | 4000   |
+-------+--------+------------+--------+
Bonus table:
+-------+-------+
| empId | bonus |
+-------+-------+
| 2     | 500   |
| 4     | 2000  |
+-------+-------+
Output: 
+------+-------+
| name | bonus |
+------+-------+
| Brad | null  |
| John | null  |
| Dan  | 500   |
+------+-------+
```

```sql
SELECT
  e.name, b.bonus
FROM Employee e
LEFT JOIN Bonus b
  ON e.empId=b.empId
WHERE bonus<1000
  OR bonus IS NULL
```

### [1280. Students and Examinations](https://leetcode.com/problems/students-and-examinations/)

Write a solution to find the number of times each student attended each exam.

Return the result table ordered by `student_id` and `subject_name`.

```
Input: 
Students table:
+------------+--------------+
| student_id | student_name |
+------------+--------------+
| 1          | Alice        |
| 2          | Bob          |
| 13         | John         |
| 6          | Alex         |
+------------+--------------+
Subjects table:
+--------------+
| subject_name |
+--------------+
| Math         |
| Physics      |
| Programming  |
+--------------+
Examinations table:
+------------+--------------+
| student_id | subject_name |
+------------+--------------+
| 1          | Math         |
| 1          | Physics      |
| 1          | Programming  |
| 2          | Programming  |
| 1          | Physics      |
| 1          | Math         |
| 13         | Math         |
| 13         | Programming  |
| 13         | Physics      |
| 2          | Math         |
| 1          | Math         |
+------------+--------------+
Output: 
+------------+--------------+--------------+----------------+
| student_id | student_name | subject_name | attended_exams |
+------------+--------------+--------------+----------------+
| 1          | Alice        | Math         | 3              |
| 1          | Alice        | Physics      | 2              |
| 1          | Alice        | Programming  | 1              |
| 2          | Bob          | Math         | 1              |
| 2          | Bob          | Physics      | 0              |
| 2          | Bob          | Programming  | 1              |
| 6          | Alex         | Math         | 0              |
| 6          | Alex         | Physics      | 0              |
| 6          | Alex         | Programming  | 0              |
| 13         | John         | Math         | 1              |
| 13         | John         | Physics      | 1              |
| 13         | John         | Programming  | 1              |
+------------+--------------+--------------+----------------+
Explanation: 
The result table should contain all students and all subjects.
Alice attended the Math exam 3 times, the Physics exam 2 times, and the Programming exam 1 time.
Bob attended the Math exam 1 time, the Programming exam 1 time, and did not attend the Physics exam.
Alex did not attend any exams.
John attended the Math exam 1 time, the Physics exam 1 time, and the Programming exam 1 time.
```



```sql
SELECT s.student_id, s.student_name, sub.subject_name, COUNT(e.subject_name) AS attended_exams
FROM Students s
CROSS JOIN Subjects sub
LEFT JOIN Examinations e ON s.student_id = e.student_id AND sub.subject_name = e.subject_name
GROUP BY s.student_id, s.student_name, sub.subject_name
ORDER BY s.student_id, sub.subject_name;
```

​		•	The CROSS JOIN ensures that every combination of students and subjects is included in the result.

​		•	This is necessary because each student should be matched with every subject, even if they did not attend any exams for that subject.

The cross-join here is to ensure that we have every unique combination of student and subject. In the event a student has not attended an exam for a particular subject, it will have a value of zero.

## Medium-level questions

### [570. Managers with at Least 5 Direct Reports](https://leetcode.com/problems/managers-with-at-least-5-direct-reports/)

Write a solution to find managers with at least five direct reports.

Return the result table in any order.

```
Input: 
Employee table:
+-----+-------+------------+-----------+
| id  | name  | department | managerId |
+-----+-------+------------+-----------+
| 101 | John  | A          | null      |
| 102 | Dan   | A          | 101       |
| 103 | James | A          | 101       |
| 104 | Amy   | A          | 101       |
| 105 | Anne  | A          | 101       |
| 106 | Ron   | B          | 101       |
+-----+-------+------------+-----------+
Output: 
+------+
| name |
+------+
| John |
+------+
```

```sql
SELECT name
FROM Employee 
WHERE id IN (SELECT managerId FROM Employee GROUP BY 1 HAVING COUNT(1) >= 5)
```

```sql
SELECT e.name
FROM Employee e 
INNER JOIN Employee m
  ON e.id=m.managerId 
GROUP BY m.managerId 
HAVING COUNT(m.managerId) >= 5
```

### [1934. Confirmation Rate](https://leetcode.com/problems/confirmation-rate/)

The confirmation rate of a user is the number of `confirmed` messages divided by the total number of requested confirmation messages. The confirmation rate of a user who did not request any confirmation messages is 0. Round the confirmation rate to two decimal places.

Write a solution to find the confirmation rate of each user.

Return the result table in any order.

```
Input: 
Signups table:
+---------+---------------------+
| user_id | time_stamp          |
+---------+---------------------+
| 3       | 2020-03-21 10:16:13 |
| 7       | 2020-01-04 13:57:59 |
| 2       | 2020-07-29 23:09:44 |
| 6       | 2020-12-09 10:39:37 |
+---------+---------------------+
Confirmations table:
+---------+---------------------+-----------+
| user_id | time_stamp          | action    |
+---------+---------------------+-----------+
| 3       | 2021-01-06 03:30:46 | timeout   |
| 3       | 2021-07-14 14:00:00 | timeout   |
| 7       | 2021-06-12 11:57:29 | confirmed |
| 7       | 2021-06-13 12:58:28 | confirmed |
| 7       | 2021-06-14 13:59:27 | confirmed |
| 2       | 2021-01-22 00:00:00 | confirmed |
| 2       | 2021-02-28 23:59:59 | timeout   |
+---------+---------------------+-----------+
Output: 
+---------+-------------------+
| user_id | confirmation_rate |
+---------+-------------------+
| 6       | 0.00              |
| 3       | 0.00              |
| 7       | 1.00              |
| 2       | 0.50              |
+---------+-------------------+
Explanation: 
User 6 did not request any confirmation messages. The confirmation rate is 0.
User 3 made 2 requests and both timed out. The confirmation rate is 0.
User 7 made 3 requests and all were confirmed. The confirmation rate is 1.
User 2 made 2 requests where one was confirmed and the other timed out. The confirmation rate is 1 / 2 = 0.5.
```

```sql
SELECT 
  s.user_id,
  ROUND(AVG(IF(c.action='confirmed', 1, 0)), 2) AS confirmation_rate
FROM Signups s
LEFT JOIN Confirmations c
  ON s.user_id=c.user_id
GROUP BY user_id
```



```sql
select s.user_id, round(avg(case when c.action = 'confirmed' then 1 else 0 END), 2) as confirmation_rate
from Signups s 
LEFT JOIN Confirmations c on s.user_id = c.user_id
group by s.user_id
# round(avg(action = 'confirmed'),2) as confirmation_rate
```

- AVG(CASE WHEN c.action = 'confirmed' THEN 1 ELSE 0 END) = AVG(action = 'confirmed')

### [1789. Primary Department for Each Employee](https://leetcode.com/problems/primary-department-for-each-employee/)

Employees can belong to multiple departments. When the employee joins other departments, they need to decide which department is their primary department. Note that when an employee belongs to only one department, their primary column is `N`.

Write a solution to report all the employees with their primary department. For employees who belong to one department, report their only department.

Return the result table in any order.

```
Input: 
Employee table:
+-------------+---------------+--------------+
| employee_id | department_id | primary_flag |
+-------------+---------------+--------------+
| 1           | 1             | N            |
| 2           | 1             | Y            |
| 2           | 2             | N            |
| 3           | 3             | N            |
| 4           | 2             | N            |
| 4           | 3             | Y            |
| 4           | 4             | N            |
+-------------+---------------+--------------+
Output: 
+-------------+---------------+
| employee_id | department_id |
+-------------+---------------+
| 1           | 1             |
| 2           | 1             |
| 3           | 3             |
| 4           | 3             |
+-------------+---------------+
Explanation: 
- The Primary department for employee 1 is 1.
- The Primary department for employee 2 is 1.
- The Primary department for employee 3 is 3.
- The Primary department for employee 4 is 3.
```

```sql
SELECT employee_id, department_id
FROM Employee
WHERE primary_flag='Y'
UNION
SELECT employee_id, department_id
FROM Employee
GROUP BY 1
HAVING COUNT(*)=1
```

The trick here is to separate employees into two buckets: those with multiple departments and those with only one.

Each bucket would have its query and resulting table. Once you have the two tables, you simply need to union them together as seen in the solution above.

### [1164. Product Price at a Given Date](https://leetcode.com/problems/product-price-at-a-given-date/)

Write a solution to find the prices of all products on 2019–08–16. Assume the price of all products before any change is 10.

Return the result table in any order.

```
Input: 
Products table:
+------------+-----------+-------------+
| product_id | new_price | change_date |
+------------+-----------+-------------+
| 1          | 20        | 2019-08-14  |
| 2          | 50        | 2019-08-14  |
| 1          | 30        | 2019-08-15  |
| 1          | 35        | 2019-08-16  |
| 2          | 65        | 2019-08-17  |
| 3          | 20        | 2019-08-18  |
+------------+-----------+-------------+
Output: 
+------------+-------+
| product_id | price |
+------------+-------+
| 2          | 50    |
| 1          | 35    |
| 3          | 10    |
+------------+-------+
```

```sql
SELECT product_id, 10 AS price
FROM Products
WHERE product_id NOT IN (SELECT DISTINCT product_id FROM Products WHERE change_date<='2019-08-16')
UNION
SELECT product_id, new_price AS price
FROM Products
WHERE (product_id, change_date) IN (SELECT product_id, MAX(change_date) FROM Products WHERE change_date<='2019-08-16' GROUP BY 1)
```

### *[1204. Last Person to Fit in the Bus](https://leetcode.com/problems/last-person-to-fit-in-the-bus/)

There is a queue of people waiting to board a bus. However, the bus has a weight limit of 1,000 kilograms, so there may be some people who cannot board.

Write a solution to find the `person_name` of the last person that can fit on the bus without exceeding the weight limit. The test cases are generated such that the first person does not exceed the weight limit.

```
Input: 
Queue table:
+-----------+-------------+--------+------+
| person_id | person_name | weight | turn |
+-----------+-------------+--------+------+
| 5         | Alice       | 250    | 1    |
| 4         | Bob         | 175    | 5    |
| 3         | Alex        | 350    | 2    |
| 6         | John Cena   | 400    | 3    |
| 1         | Winston     | 500    | 6    |
| 2         | Marie       | 200    | 4    |
+-----------+-------------+--------+------+
Output: 
+-------------+
| person_name |
+-------------+
| John Cena   |
+-------------+
Explanation: The folowing table is ordered by the turn for simplicity.
+------+----+-----------+--------+--------------+
| Turn | ID | Name      | Weight | Total Weight |
+------+----+-----------+--------+--------------+
| 1    | 5  | Alice     | 250    | 250          |
| 2    | 3  | Alex      | 350    | 600          |
| 3    | 6  | John Cena | 400    | 1000         | (last person to board)
| 4    | 2  | Marie     | 200    | 1200         | (cannot board)
| 5    | 4  | Bob       | 175    | ___          |
| 6    | 1  | Winston   | 500    | ___          |
+------+----+-----------+--------+--------------+

```

```sql
SELECT q1.person_name
FROM Queue q1
LEFT JOIN Queue q2
  ON q1.turn >= q2.turn
GROUP BY q1.turn
HAVING SUM(q2.weight) <= 1000
ORDER BY SUM(q2.weight) DESC
LIMIT 1
```

For each turn, get the weights of all the preceding turns and the current turn. If the sum of all preceding turns and current turn is less than or equal to a thousand, then return the passenger name.

### [1907. Count Salary Categories](https://leetcode.com/problems/count-salary-categories/)

Write a solution to calculate the number of bank accounts for each salary category. The salary categories are

- `Low Salary`: All the salaries strictly less than 20,000
- `Average Salary`: All the salaries in the inclusive range [20,000, 50,000]
- `High Salary`: All the salaries strictly greater than 50,000

The result table must contain all three categories. If there are no accounts in a category, return 0.

Return the result table in any order.

```
Input: 
Accounts table:
+------------+--------+
| account_id | income |
+------------+--------+
| 3          | 108939 |
| 2          | 12747  |
| 8          | 87709  |
| 6          | 91796  |
+------------+--------+
Output: 
+----------------+----------------+
| category       | accounts_count |
+----------------+----------------+
| Low Salary     | 1              |
| Average Salary | 0              |
| High Salary    | 3              |
+----------------+----------------+
Explanation: 
Low Salary: Account 2.
Average Salary: No accounts.
High Salary: Accounts 3, 6, and 8.

```

```sql
SELECT 'Low Salary' AS category, SUM(income<20000) AS accounts_count
FROM Accounts
UNION
SELECT 'Average Salary' AS category, SUM(income>=20000 AND income<=50000) AS accounts_count
FROM Accounts
UNION
SELECT 'High Salary' AS category, SUM(income>50000) AS accounts_count
FROM Accounts
```

- The reason a union is needed here and not a simple case when followed by a group by **count** is because we want t**he category without any accounts to still return a value of zero**.

Similar to the employee department question above, the trick here is to split up your query into two parts: one for products with price change before 2019–08–16, and one for after.

For products with price changes after 2019–08–16, set the default price as 10. On the other hand, for products with price change before 2019–08–16, return the most recent price.

## Hard-level questions 

## [Highest Cost orders](https://platform.stratascratch.com/coding/9915-highest-cost-orders?python=&code_type=3)

![image-20241120120459663](/images/2024-11-19-SQL_stratascratch/image-20241120120459663.png)

![image-20241120120528464](/images/2024-11-19-SQL_stratascratch/image-20241120120528464.png)

![image-20241120120538244](/images/2024-11-19-SQL_stratascratch/image-20241120120538244.png)

```sql
SELECT first_name,
       sum(total_order_cost) AS total_order_cost,
       order_date
FROM orders o
LEFT JOIN customers c ON o.cust_id = c.id
WHERE order_date BETWEEN '2019-02-1' AND '2019-05-1'
GROUP BY first_name,
         order_date
HAVING sum(total_order_cost) =
  (SELECT max(total_order_cost)
   FROM
     (SELECT sum(total_order_cost) AS total_order_cost
      FROM orders
      WHERE order_date BETWEEN '2019-02-1' AND '2019-05-1'
      GROUP BY cust_id,
               order_date) b);
```

- The query uses the SUM() aggregate function to calculate the total order cost. To get all data you need, you have to LEFT JOIN two tables. Data is filtered on order date using the WHERE clause.
- Next, data is grouped by the customer’s first name and order date. You need to output the customer with the highest daily total order. To do that, you need the HAVING clause to get data where the sum of the order costs per customer and per date is equal to the order maximum. This is where you need another aggregate function, which is MAX().

# Basic Aggregate Functions

## Easy questions

### [620. Not Boring Movies](https://leetcode.com/problems/not-boring-movies/)

Write a solution to report the movies with an odd-numbered ID and a description that is not `boring`.

Return the result table ordered by `rating` in descending order.

```
Input: 
Cinema table:
+----+------------+-------------+--------+
| id | movie      | description | rating |
+----+------------+-------------+--------+
| 1  | War        | great 3D    | 8.9    |
| 2  | Science    | fiction     | 8.5    |
| 3  | irish      | boring      | 6.2    |
| 4  | Ice song   | Fantacy     | 8.6    |
| 5  | House card | Interesting | 9.1    |
+----+------------+-------------+--------+
Output: 
+----+------------+-------------+--------+
| id | movie      | description | rating |
+----+------------+-------------+--------+
| 5  | House card | Interesting | 9.1    |
| 1  | War        | great 3D    | 8.9    |
+----+------------+-------------+--------+
Explanation: 
We have three movies with odd-numbered IDs: 1, 3, and 5. The movie with ID = 3 is boring so we do not include it in the answer.

```

```sql
SELECT *
FROM Cinema
WHERE MOD(id, 2) = 1 # id % 2 = 1
  AND description <> 'boring' 
ORDER BY rating DESC
```

### [1251. Average Selling Price](https://leetcode.com/problems/average-selling-price/)

Write a solution to find the average selling price for each product. `average_price` should be rounded to two decimal places.

Return the result table in any order.

```
Input: 
Prices table:
+------------+------------+------------+--------+
| product_id | start_date | end_date   | price  |
+------------+------------+------------+--------+
| 1          | 2019-02-17 | 2019-02-28 | 5      |
| 1          | 2019-03-01 | 2019-03-22 | 20     |
| 2          | 2019-02-01 | 2019-02-20 | 15     |
| 2          | 2019-02-21 | 2019-03-31 | 30     |
+------------+------------+------------+--------+
UnitsSold table:
+------------+---------------+-------+
| product_id | purchase_date | units |
+------------+---------------+-------+
| 1          | 2019-02-25    | 100   |
| 1          | 2019-03-01    | 15    |
| 2          | 2019-02-10    | 200   |
| 2          | 2019-03-22    | 30    |
+------------+---------------+-------+
Output: 
+------------+---------------+
| product_id | average_price |
+------------+---------------+
| 1          | 6.96          |
| 2          | 16.96         |
+------------+---------------+
Explanation: 
Average selling price = Total Price of Product / Number of products sold.
Average selling price for product 1 = ((100 * 5) + (15 * 20)) / 115 = 6.96
Average selling price for product 2 = ((200 * 15) + (30 * 30)) / 230 = 16.96
```

```sql
SELECT 
  p.product_id
  , COALESCE(round(SUM(units*price)/SUM(units), 2), 0) AS average_price
FROM Prices p
LEFT JOIN UnitsSold u
  ON p.product_id=u.product_id
  AND u.purchase_date BETWEEN p.start_date AND p.end_date
GROUP BY 1
```

### [1633. Percentage of Users Attended a Contest](https://leetcode.com/problems/percentage-of-users-attended-a-contest/)

Write a solution to find the percentage of the users registered in each contest rounded to two decimals.

Return the result table ordered by `percentage` in descending order. In case of a tie, order it by `contest_id` in ascending order.

```
Input: 
Users table:
+---------+-----------+
| user_id | user_name |
+---------+-----------+
| 6       | Alice     |
| 2       | Bob       |
| 7       | Alex      |
+---------+-----------+
Register table:
+------------+---------+
| contest_id | user_id |
+------------+---------+
| 215        | 6       |
| 209        | 2       |
| 208        | 2       |
| 210        | 6       |
| 208        | 6       |
| 209        | 7       |
| 209        | 6       |
| 215        | 7       |
| 208        | 7       |
| 210        | 2       |
| 207        | 2       |
| 210        | 7       |
+------------+---------+
Output: 
+------------+------------+
| contest_id | percentage |
+------------+------------+
| 208        | 100.0      |
| 209        | 100.0      |
| 210        | 100.0      |
| 215        | 66.67      |
| 207        | 33.33      |
+------------+------------+
Explanation: 
All the users registered in contests 208, 209, and 210. The percentage is 100% and we sort them in the answer table by contest_id in ascending order.
Alice and Alex registered in contest 215 and the percentage is ((2/3) * 100) = 66.67%
Bob registered in contest 207 and the percentage is ((1/3) * 100) = 33.33%
```

```sql
# Write your MySQL query statement below
select contest_id, round( count(user_id)/ (select count(user_id) from Users)*100, 2) as percentage
from register
group by contest_id
order by percentage desc, contest_id
```

```sql
SELECT 
  contest_id
  , ROUND(100*COUNT(1)/(SELECT COUNT(DISTINCT user_id) FROM Users), 2) AS percentage # count(1) = count(*)
FROM Register
GROUP BY contest_id
ORDER BY percentage DESC, contest_id
```

### [1211. Queries Quality and Percentage](https://leetcode.com/problems/queries-quality-and-percentage/)

We define query `quality` as the average of the ratio between query rating and its position. We also define `poor_query_percentage` as the percentage of all queries with rating less than three.

Both `quality` and `poor_query_percentage` should be rounded to two decimal places.

Return the result table in any order.

```
Input: 
Queries table:
+------------+-------------------+----------+--------+
| query_name | result            | position | rating |
+------------+-------------------+----------+--------+
| Dog        | Golden Retriever  | 1        | 5      |
| Dog        | German Shepherd   | 2        | 5      |
| Dog        | Mule              | 200      | 1      |
| Cat        | Shirazi           | 5        | 2      |
| Cat        | Siamese           | 3        | 3      |
| Cat        | Sphynx            | 7        | 4      |
+------------+-------------------+----------+--------+
Output: 
+------------+---------+-----------------------+
| query_name | quality | poor_query_percentage |
+------------+---------+-----------------------+
| Dog        | 2.50    | 33.33                 |
| Cat        | 0.66    | 33.33                 |
+------------+---------+-----------------------+
Explanation: 
Dog queries quality is ((5 / 1) + (5 / 2) + (1 / 200)) / 3 = 2.50
Dog queries poor_ query_percentage is (1 / 3) * 100 = 33.33

Cat queries quality equals ((2 / 5) + (3 / 3) + (4 / 7)) / 3 = 0.66
Cat queries poor_ query_percentage is (1 / 3) * 100 = 33.33
```

```sql
SELECT 
  query_name
  , ROUND(AVG(rating/position), 2) AS quality
  , ROUND(100*AVG(IF(rating<3, 1, 0)), 2) AS poor_query_percentage #if만들때 주의!
FROM Queries
WHERE query_name IS NOT NULL 
GROUP BY 1
```

## Medium questions

### [1193. Monthly Transactions I](https://leetcode.com/problems/monthly-transactions-i/)

Write an SQL query to find for each month and country, the number of transactions and their total amount, the number of approved transactions and their total amount.

Return the result table in any order.

```
Input: 
Transactions table:
+------+---------+----------+--------+------------+
| id   | country | state    | amount | trans_date |
+------+---------+----------+--------+------------+
| 121  | US      | approved | 1000   | 2018-12-18 |
| 122  | US      | declined | 2000   | 2018-12-19 |
| 123  | US      | approved | 2000   | 2019-01-01 |
| 124  | DE      | approved | 2000   | 2019-01-07 |
+------+---------+----------+--------+------------+
Output: 
+----------+---------+-------------+----------------+--------------------+-----------------------+
| month    | country | trans_count | approved_count | trans_total_amount | approved_total_amount |
+----------+---------+-------------+----------------+--------------------+-----------------------+
| 2018-12  | US      | 2           | 1              | 3000               | 1000                  |
| 2019-01  | US      | 1           | 1              | 2000               | 2000                  |
| 2019-01  | DE      | 1           | 1              | 2000               | 2000                  |
+----------+---------+-------------+----------------+--------------------+-----------------------+
```

```sql
SELECT 
  date_format(trans_date, '%Y-%m') AS month
  , country
  , COUNT(1) AS trans_count
  , SUM(IF(state='approved', 1, 0)) AS approved_count
  # sum(case when state = 'approved' then 1 else 0 end )
  , SUM(amount) as trans_total_amount
  , SUM(IF(state='approved', amount, 0)) AS approved_total_amount
FROM Transactions
GROUP BY 1, 2
```

### [*1174. Immediate Food Delivery II](https://leetcode.com/problems/immediate-food-delivery-ii/)

If the customer’s preferred delivery date is the same as the order date, then the order is called immediate. Otherwise, it is called scheduled.

The first order of a customer is the order with the earliest order date that the customer made. It is guaranteed that a customer has precisely one first order.

Write a solution to find the percentage of immediate orders in the first orders of all customers, rounded to two decimal places.

```
Input: 
Delivery table:
+-------------+-------------+------------+-----------------------------+
| delivery_id | customer_id | order_date | customer_pref_delivery_date |
+-------------+-------------+------------+-----------------------------+
| 1           | 1           | 2019-08-01 | 2019-08-02                  |
| 2           | 2           | 2019-08-02 | 2019-08-02                  |
| 3           | 1           | 2019-08-11 | 2019-08-12                  |
| 4           | 3           | 2019-08-24 | 2019-08-24                  |
| 5           | 3           | 2019-08-21 | 2019-08-22                  |
| 6           | 2           | 2019-08-11 | 2019-08-13                  |
| 7           | 4           | 2019-08-09 | 2019-08-09                  |
+-------------+-------------+------------+-----------------------------+
Output: 
+----------------------+
| immediate_percentage |
+----------------------+
| 50.00                |
+----------------------+
Explanation: 
The customer id 1 has a first order with delivery id 1 and it is scheduled.
The customer id 2 has a first order with delivery id 2 and it is immediate.
The customer id 3 has a first order with delivery id 5 and it is scheduled.
The customer id 4 has a first order with delivery id 7 and it is immediate.
Hence, half the customers have immediate first orders.
```

```sql
SELECT ROUND(100*AVG(order_date=customer_pref_delivery_date), 2) AS immediate_percentage
FROM Delivery
WHERE (customer_id, order_date) IN (SELECT customer_id, MIN(order_date) FROM Delivery GROUP BY 1)
```

 ### [550. Game Play Analysis IV](https://leetcode.com/problems/game-play-analysis-iv/)

Write a solution to report the fraction of players that logged in again on the day after the day they first logged in, rounded to two decimal places. In other words, you need to count the number of players that logged in for at least two consecutive days starting from their first login date, then divide that number by the total number of players.

```
Input: 
Activity table:
+-----------+-----------+------------+--------------+
| player_id | device_id | event_date | games_played |
+-----------+-----------+------------+--------------+
| 1         | 2         | 2016-03-01 | 5            |
| 1         | 2         | 2016-03-02 | 6            |
| 2         | 3         | 2017-06-25 | 1            |
| 3         | 1         | 2016-03-02 | 0            |
| 3         | 4         | 2018-07-03 | 5            |
+-----------+-----------+------------+--------------+
Output: 
+-----------+
| fraction  |
+-----------+
| 0.33      |
+-----------+
Explanation: 
Only the player with id 1 logged back in after the first day he had logged in so the answer is 1/3 = 0.33
```

```sql
SELECT ROUND(COUNT(DISTINCT player_id)/(SELECT COUNT(DISTINCT player_id) FROM Activity), 2) AS fraction
FROM Activity
WHERE (player_id, date_sub(event_date, INTERVAL 1 DAY)) IN (SELECT player_id, MIN(event_date) FROM Activity GROUP BY 1)
```

# WHERE Clause

## [Top Cool Votes](https://platform.stratascratch.com/coding/10060-top-cool-votes?python=&code_type=3)

![image-20241119182706852](/images/2024-11-19-SQL_stratascratch/image-20241119182706852.png)

![image-20241119182804544](/images/2024-11-19-SQL_stratascratch/image-20241119182804544.png)

```sql
select business_name, review_text
from yelp_reviews
where cool = 
    (select max(cool)
     From yelp_reviews);
```

## [The Most Popular Client_Id Among Users Using Video and Voice Calls](https://platform.stratascratch.com/coding/2029-the-most-popular-client_id-among-users-using-video-and-voice-calls?python=&code_type=3)

![image-20241119193308392](/images/2024-11-19-SQL_stratascratch/image-20241119193308392.png)

```sql
-- users who have at least 50% of their events being call events
-- count these users per client_id
-- limit output to 1 client_id (most users)
SELECT client_id
FROM

(SELECT client_id, user_id
FROM fact_events
GROUP BY 1, 2
HAVING AVG(CASE WHEN event_type IN ('video call received', 'video call sent', 'voice call received', 'voice call sent') THEN 1 ELSE 0 END) >= 0.5) call_users

GROUP BY 1
ORDER BY COUNT(user_id) DESC
LIMIT 1
```

# Sorting and Grouping

## Easy questions

### [1141. User Activity for the Past 30 Days I](https://leetcode.com/problems/user-activity-for-the-past-30-days-i/)

Write a solution to find the daily active user count for a period of 30 days ending 2019–07–27 inclusively. A user was active on someday if they made at least one activity on that day.

Return the result table in any order.

```
Input: 
Activity table:
+---------+------------+---------------+---------------+
| user_id | session_id | activity_date | activity_type |
+---------+------------+---------------+---------------+
| 1       | 1          | 2019-07-20    | open_session  |
| 1       | 1          | 2019-07-20    | scroll_down   |
| 1       | 1          | 2019-07-20    | end_session   |
| 2       | 4          | 2019-07-20    | open_session  |
| 2       | 4          | 2019-07-21    | send_message  |
| 2       | 4          | 2019-07-21    | end_session   |
| 3       | 2          | 2019-07-21    | open_session  |
| 3       | 2          | 2019-07-21    | send_message  |
| 3       | 2          | 2019-07-21    | end_session   |
| 4       | 3          | 2019-06-25    | open_session  |
| 4       | 3          | 2019-06-25    | end_session   |
+---------+------------+---------------+---------------+
Output: 
+------------+--------------+ 
| day        | active_users |
+------------+--------------+ 
| 2019-07-20 | 2            |
| 2019-07-21 | 2            |
+------------+--------------+ 
Explanation: Note that we do not care about days with zero active users.
```

```sql
SELECT 
  activity_date AS day,
  COUNT(DISTINCT user_id) AS active_users
FROM Activity
WHERE activity_date BETWEEN '2019-06-28' AND '2019-07-27'
GROUP BY 1
```

- 1. (distinct activity_date) **is incorrect**: **Fix**: Remove distinct since activity_date is already being grouped in GROUP BY.

  2. **Incorrect placement of** WHERE **clause**

     ​	•**Fix**: WHERE comes **before** GROUP BY, not after.

## Medium questions

### [1045. Customers Who Bought All Products](https://leetcode.com/problems/customers-who-bought-all-products/)

Write a solution to report the customer ids from the `Customer` table that bought all the products in the `Product` table.

Return the result table in any order.

```
Input: 
Customer table:
+-------------+-------------+
| customer_id | product_key |
+-------------+-------------+
| 1           | 5           |
| 2           | 6           |
| 3           | 5           |
| 3           | 6           |
| 1           | 6           |
+-------------+-------------+
Product table:
+-------------+
| product_key |
+-------------+
| 5           |
| 6           |
+-------------+
Output: 
+-------------+
| customer_id |
+-------------+
| 1           |
| 3           |
+-------------+
Explanation: 
The customers who bought all the products (5 and 6) are customers with IDs 1 and 3.
```

```sql
SELECT customer_id
FROM Customer 
GROUP BY 1
HAVING COUNT(DISTINCT product_key)=(SELECT COUNT(*) FROM Product)
```

### [3 Bed Minimum](https://platform.stratascratch.com/coding/9627-3-bed-minimum?python=&code_type=3)

![image-20241119202919929](/images/2024-11-19-SQL_stratascratch/image-20241119202919929.png)

```sql
-- neighborhood name, 
-- the average number of beds, descending order
-- at least 3 beds in total
SELECT neighbourhood, AVG(beds)
FROM airbnb_search_details
WHERE neighbourhood IS NOT NULL
GROUP BY neighbourhood
HAVING SUM(beds) > 3
ORDER BY AVG(beds) DESC;
```

# Ranking Rows and LIMIT Clause

## [Ranking Most Active Guests](https://platform.stratascratch.com/coding/10159-ranking-most-active-guests?python=&code_type=3)

![image-20241119232801316](/images/2024-11-19-SQL_stratascratch/image-20241119232801316.png)

![image-20241119232821804](/images/2024-11-19-SQL_stratascratch/image-20241119232821804.png)

```sql
-- Rank guests, the total number of messages
-- same messages - same rank
-- output: rank, guest id, # of total messages they've sent.
-- order by desc.
SELECT 
    DENSE_RANK() OVER(ORDER BY sum(n_messages) DESC) as ranking, 
    id_guest, 
    sum(n_messages) as sum_n_messages
FROM airbnb_contacts
GROUP BY id_guest
ORDER BY sum_n_messages; -- Desc is unnecessary since the ranks are already generated in descending order.
```

## [Most checkins](https://platform.stratascratch.com/coding/10053-most-checkins?python=&code_type=3)

![image-20241119233540525](/images/2024-11-19-SQL_stratascratch/image-20241119233540525.png)

![image-20241119233557315](/images/2024-11-19-SQL_stratascratch/image-20241119233557315.png)

```sql
-- top 5 business with the most check-ins
-- business id, # of check-ins
SELECT 
    business_id,
    sum(checkins) AS n_checkins
FROM yelp_checkin
GROUP BY 
    business_id
ORDER BY
    n_checkins DESC
LIMIT 5;
```

# Subqueries and CTEs

## Easy question 

### *[1978. Employees Whose Manager Left the Company](https://leetcode.com/problems/employees-whose-manager-left-the-company/)

Find the IDs of the employees whose salary is strictly less than $30,000 and whose manager left the company. When a manager leaves the company, their information is deleted from the `Employees` table, but the reports still have their `manager_id` set to the manager that left.

Return the result table ordered by `employee_id`.

```
Input:  
Employees table:
+-------------+-----------+------------+--------+
| employee_id | name      | manager_id | salary |
+-------------+-----------+------------+--------+
| 3           | Mila      | 9          | 60301  |
| 12          | Antonella | null       | 31000  |
| 13          | Emery     | null       | 67084  |
| 1           | Kalel     | 11         | 21241  |
| 9           | Mikaela   | null       | 50937  |
| 11          | Joziah    | 6          | 28485  |
+-------------+-----------+------------+--------+
Output: 
+-------------+
| employee_id |
+-------------+
| 11          |
+-------------+
Explanation: 
The employees with a salary less than $30000 are 1 (Kalel) and 11 (Joziah).
Kalel's manager is employee 11, who is still in the company (Joziah).
Joziah's manager is employee 6, who left the company because there is no row for employee 6 as it was deleted.
```

```sql
SELECT employee_id
FROM Employees 
WHERE salary<30000
  AND manager_id NOT IN (SELECT DISTINCT employee_id FROM Employees)
```

- manager_id NOT IN employee_id is incorrect because employee_id is **not** a list.
  - ✅ **Fix**: Use a **subquery** (SELECT employee_id FROM Employees) to check if manager_id exists in Employees.

## Medium questions

### [626. Exchange Seats](https://leetcode.com/problems/exchange-seats/)

Write a solution to swap the seat id of every two consecutive students. If the number of students is odd, the id of the last student is not swapped.

Return the result table ordered by `id` in ascending order.

```
Input: 
Seat table:
+----+---------+
| id | student |
+----+---------+
| 1  | Abbot   |
| 2  | Doris   |
| 3  | Emerson |
| 4  | Green   |
| 5  | Jeames  |
+----+---------+
Output: 
+----+---------+
| id | student |
+----+---------+
| 1  | Doris   |
| 2  | Abbot   |
| 3  | Green   |
| 4  | Emerson |
| 5  | Jeames  |
+----+---------+
Explanation: 
Note that if the number of students is odd, there is no need to change the last one's seat.
```

```sql
WITH cte AS (
  SELECT 
    *
    , LAG(student) OVER() AS prev_student
    , LEAD(student) OVER() AS next_student
  FROM Seat
)
SELECT 
  id
  , CASE WHEN MOD(id, 2)=1 AND next_student IS NOT NULL THEN next_student
         WHEN MOD(id, 2)=0 THEN prev_student
         ELSE student
    END AS student
FROM cte
```

For each ID, get the previous and next student if any. If the ID is odd and there is a next student, swap places with the next student. If the ID is even, then swap places with the previous student. Otherwise, the remaining ID has to be the last student in an odd-numbered group, in which case the student remains in their original place.

```sql
SELECT
  ROW_NUMBER() OVER() AS id
  , student
FROM seat
ORDER BY IF(MOD(id, 2)=0, id-1, id+1)
```

There is also a more algorithmically elegant way to do what we just did, which at first glance, may seem a little difficult to comprehend. Let’s break this down.

First, if the original ID is even, then minus one and conversely, if the original ID is odd, then plus one. We want to then sort the table based on the newly created IDs which indirectly swaps two consecutive seats.

But now that the original IDs are swapped, how do we make sure they are still sorted based on the new student order? This is where we use the row number window function to re-assign the new IDs.

### [1341. Movie Rating](https://leetcode.com/problems/movie-rating/)

Write a solution to:

- Find the name of the user who has rated the greatest number of movies. In case of a tie, return the lexicographically smaller user name.
- Find the movie name with the highest average rating in February 2020. In case of a tie, return the lexicographically smaller movie name.

```
Input: 
Movies table:
+-------------+--------------+
| movie_id    |  title       |
+-------------+--------------+
| 1           | Avengers     |
| 2           | Frozen 2     |
| 3           | Joker        |
+-------------+--------------+
Users table:
+-------------+--------------+
| user_id     |  name        |
+-------------+--------------+
| 1           | Daniel       |
| 2           | Monica       |
| 3           | Maria        |
| 4           | James        |
+-------------+--------------+
MovieRating table:
+-------------+--------------+--------------+-------------+
| movie_id    | user_id      | rating       | created_at  |
+-------------+--------------+--------------+-------------+
| 1           | 1            | 3            | 2020-01-12  |
| 1           | 2            | 4            | 2020-02-11  |
| 1           | 3            | 2            | 2020-02-12  |
| 1           | 4            | 1            | 2020-01-01  |
| 2           | 1            | 5            | 2020-02-17  | 
| 2           | 2            | 2            | 2020-02-01  | 
| 2           | 3            | 2            | 2020-03-01  |
| 3           | 1            | 3            | 2020-02-22  | 
| 3           | 2            | 4            | 2020-02-25  | 
+-------------+--------------+--------------+-------------+
Output: 
+--------------+
| results      |
+--------------+
| Daniel       |
| Frozen 2     |
+--------------+
Explanation: 
Daniel and Monica have rated 3 movies ("Avengers", "Frozen 2" and "Joker") but Daniel is smaller lexicographically.
Frozen 2 and Joker have a rating average of 3.5 in February but Frozen 2 is smaller lexicographically.

```

```sql
(SELECT u.name AS results
FROM MovieRating r
LEFT JOIN Users u
  ON r.user_id=u.user_id
GROUP BY u.user_id 
ORDER BY count(*) DESC, u.name
LIMIT 1)
UNION ALL 
(SELECT m.title AS results 
FROM MovieRating r
LEFT JOIN Movies m
  ON r.movie_id=m.movie_id
WHERE r.created_at like '2020-02%'
GROUP BY r.movie_id
ORDER BY AVG(r.rating) DESC, m.title
LIMIT 1)
```

### [1321. Restaurant Growth](https://leetcode.com/problems/restaurant-growth/)

You are the restaurant owner and you want to analyze a possible expansion (there will be at least one customer every day).

Compute the moving average of how much the customer paid in a seven-day window (current day plus six days before). `average_amount` should be rounded to two decimal places.

Return the result table ordered by `visited_on` in ascending order.

```
Input: 
Customer table:
+-------------+--------------+--------------+-------------+
| customer_id | name         | visited_on   | amount      |
+-------------+--------------+--------------+-------------+
| 1           | Jhon         | 2019-01-01   | 100         |
| 2           | Daniel       | 2019-01-02   | 110         |
| 3           | Jade         | 2019-01-03   | 120         |
| 4           | Khaled       | 2019-01-04   | 130         |
| 5           | Winston      | 2019-01-05   | 110         | 
| 6           | Elvis        | 2019-01-06   | 140         | 
| 7           | Anna         | 2019-01-07   | 150         |
| 8           | Maria        | 2019-01-08   | 80          |
| 9           | Jaze         | 2019-01-09   | 110         | 
| 1           | Jhon         | 2019-01-10   | 130         | 
| 3           | Jade         | 2019-01-10   | 150         | 
+-------------+--------------+--------------+-------------+
Output: 
+--------------+--------------+----------------+
| visited_on   | amount       | average_amount |
+--------------+--------------+----------------+
| 2019-01-07   | 860          | 122.86         |
| 2019-01-08   | 840          | 120            |
| 2019-01-09   | 840          | 120            |
| 2019-01-10   | 1000         | 142.86         |
+--------------+--------------+----------------+
Explanation: 
1st moving average from 2019-01-01 to 2019-01-07 has an average_amount of (100 + 110 + 120 + 130 + 110 + 140 + 150)/7 = 122.86
2nd moving average from 2019-01-02 to 2019-01-08 has an average_amount of (110 + 120 + 130 + 110 + 140 + 150 + 80)/7 = 120
3rd moving average from 2019-01-03 to 2019-01-09 has an average_amount of (120 + 130 + 110 + 140 + 150 + 80 + 110)/7 = 120
4th moving average from 2019-01-04 to 2019-01-10 has an average_amount of (130 + 110 + 140 + 150 + 80 + 110 + 130 + 150)/7 = 142.86
```

```sql
SELECT
  a.visited_on AS visited_on
  , SUM(b.day_sum) AS amount
  , ROUND(AVG(b.day_sum), 2) AS average_amount
FROM
  (SELECT visited_on, SUM(amount) AS day_sum FROM Customer GROUP BY visited_on) a
  , (SELECT visited_on, SUM(amount) AS day_sum FROM Customer GROUP BY visited_on) b
WHERE DATEDIFF(a.visited_on, b.visited_on) BETWEEN 0 AND 6
GROUP BY a.visited_on
HAVING COUNT(b.visited_on) = 7
```

Use inner join to get the preceding six days, if applicable, for each visit date and calculate the average of the seven days total (the current day plus the six preceding days).

### [602. Friend Requests II: Who Has the Most Friends](https://leetcode.com/problems/friend-requests-ii-who-has-the-most-friends/)

Write a solution to find the people who have the most friends and the most friends number.

```
Input: 
RequestAccepted table:
+--------------+-------------+-------------+
| requester_id | accepter_id | accept_date |
+--------------+-------------+-------------+
| 1            | 2           | 2016/06/03  |
| 1            | 3           | 2016/06/08  |
| 2            | 3           | 2016/06/08  |
| 3            | 4           | 2016/06/09  |
+--------------+-------------+-------------+
Output: 
+----+-----+
| id | num |
+----+-----+
| 3  | 3   |
+----+-----+
Explanation: 
The person with id 3 is a friend of people 1, 2, and 4, so he has three friends in total, which is the most number than any others.
```

```sql
WITH cte AS (
  SELECT requester_id AS id FROM RequestAccepted
  UNION ALL 
  SELECT accepter_id id FROM RequestAccepted
)
SELECT id, COUNT(*) AS num
FROM cte
GROUP BY 1
ORDER BY 2 DESC
LIMIT 1
```

Every ID has the same number of friends as they are the number of times they appear as a row in the table. A person forms a new friend by either sending or accepting a request.

Once you learn this pattern, this problem should be very easy to crack.

### [585. Investments in 2016](https://leetcode.com/problems/investments-in-2016/)

Write a solution to report the sum of all total investment values in 2016 `tiv_2016`, for all policyholders who:

- Have the same `tiv_2015` value as one or more other policyholders
- Are not located in the same city as any other policyholder (`lat, lon` attribute pairs must be unique)

```
Input: 
Insurance table:
+-----+----------+----------+-----+-----+
| pid | tiv_2015 | tiv_2016 | lat | lon |
+-----+----------+----------+-----+-----+
| 1   | 10       | 5        | 10  | 10  |
| 2   | 20       | 20       | 20  | 20  |
| 3   | 10       | 30       | 20  | 20  |
| 4   | 10       | 40       | 40  | 40  |
+-----+----------+----------+-----+-----+
Output: 
+----------+
| tiv_2016 |
+----------+
| 45.00    |
+----------+
Explanation: 
The first record in the table, like the last record, meets both of the two criteria.
The tiv_2015 value 10 is the same as the third and fourth records, and its location is unique.
The second record does not meet any of the two criteria. Its tiv_2015 is not like any other policyholders and its location is the same as the third record, which makes the third record fail, too.
So, the result is the sum of tiv_2016 of the first and last record, which is 45.
WITH cte AS (
  SELECT 
    *
    , COUNT(*) OVER(PARTITION BY tiv_2015) AS cnt1
    , COUNT(*) OVER(PARTITION BY lat, lon) AS cnt2
  FROM Insurance
)
SELECT ROUND(SUM(tiv_2016), 2) AS tiv_2016
FROM cte 
WHERE cnt1>1 AND cnt2=1
```

### [Top 10 Songs](https://platform.stratascratch.com/coding/9743-top-10-songs?python=&code_type=3)

![image-20241120104037030](/../images/2024-11-19-SQL_stratascratch/image-20241120104037030.png)

![image-20241120104050814](/images/2024-11-19-SQL_stratascratch/image-20241120104050814.png)

```sql
-- # of songs of each artist, the top 10 over the years.
-- order based on # of top 10 ranked songs in desc.
SELECT 
    artist, 
    count(distinct song_name) AS top10_songs_count
FROM
    (SELECT 
        artist,
        song_name
     FROM billboard_top_100_year_end
     WHERE
        year_rank <= 10
    ) temporary
GROUP BY
    artist
ORDER BY
    top10_songs_count DESC;
```

### [Find the top 5 cities with the most 5-star businesses](https://platform.stratascratch.com/coding/10148-find-the-top-10-cities-with-the-most-5-star-businesses?python=&code_type=3)

![image-20241120113756410](/../images/2024-11-19-SQL_stratascratch/image-20241120113756410.png)

![image-20241120113816176](/images/2024-11-19-SQL_stratascratch/image-20241120113816176.png)

```sql
-- the top 5 cities, highest number of 5 star businesses
-- output: city name, # of the total count of 5 star businesses, open and closed
WITH cte_5_stars AS
  (SELECT city,
          count(*) AS count_of_5_stars,
          rank() over(
                      ORDER BY count(*) DESC) AS rnk
   FROM yelp_business
   WHERE stars = 5
   GROUP BY 1)
SELECT city,
       count_of_5_stars
FROM cte_5_stars
WHERE rnk <= 5
ORDER BY count_of_5_stars DESC;
```

## Hard questions

### [185. Department Top Three Salaries](https://leetcode.com/problems/department-top-three-salaries/)

A company’s executives are interested in seeing who earns the most money in each of the company’s departments. A high earner in a department is an employee who has a salary in the top three unique salaries for that department.

Write a solution to find the employees who are high earners in each of the departments.

Return the result table in any order.

```
Input: 
Employee table:
+----+-------+--------+--------------+
| id | name  | salary | departmentId |
+----+-------+--------+--------------+
| 1  | Joe   | 85000  | 1            |
| 2  | Henry | 80000  | 2            |
| 3  | Sam   | 60000  | 2            |
| 4  | Max   | 90000  | 1            |
| 5  | Janet | 69000  | 1            |
| 6  | Randy | 85000  | 1            |
| 7  | Will  | 70000  | 1            |
+----+-------+--------+--------------+
Department table:
+----+-------+
| id | name  |
+----+-------+
| 1  | IT    |
| 2  | Sales |
+----+-------+
Output: 
+------------+----------+--------+
| Department | Employee | Salary |
+------------+----------+--------+
| IT         | Max      | 90000  |
| IT         | Joe      | 85000  |
| IT         | Randy    | 85000  |
| IT         | Will     | 70000  |
| Sales      | Henry    | 80000  |
| Sales      | Sam      | 60000  |
+------------+----------+--------+
Explanation: 
In the IT department:
- Max earns the highest unique salary
- Both Randy and Joe earn the second-highest unique salary
- Will earns the third-highest unique salary
In the Sales department:
- Henry earns the highest salary
- Sam earns the second-highest salary
- There is no third-highest salary as there are only two employees

```

```sql
WITH cte AS (
  SELECT
    e.id
    , e.name AS Employee
    , e.salary AS Salary
    , DENSE_RANK() OVER(PARTITION BY d.name ORDER BY e.salary DESC) AS dept_rank
    , d.name AS Department
  FROM Employee e
  LEFT JOIN Department d
    ON e.departmentId=d.id
)
SELECT
  Department
  , Employee
  , Salary
FROM cte
WHERE dept_rank<=3
```

#  Data Organizing and Pattern Matching

## [Classify Business Type](https://platform.stratascratch.com/coding/9726-classify-business-type?python=&code_type=3)

![image-20241120122017927](/images/2024-11-19-SQL_stratascratch/image-20241120122017927.png)

![image-20241120122052679](/images/2024-11-19-SQL_stratascratch/image-20241120122052679.png)

```sql
SELECT distinct business_name,
       CASE
           WHEN lower(business_name) like '%school%' THEN 'school'
           WHEN lower(business_name) like '%restaurant%' THEN 'restaurant'
           WHEN lower(business_name) like '%cafe%' or '%café%' or '%coffee%' THEN 'cafe'
           ELSE 'other'
       END AS business_type
FROM sf_restaurant_health_violations;
```

## [Liking Score Rating](https://platform.stratascratch.com/coding/9775-liking-score-rating?python=&code_type=3)

![image-20241120122829551](/images/2024-11-19-SQL_stratascratch/image-20241120122829551.png)

![image-20241120122906714](/images/2024-11-19-SQL_stratascratch/image-20241120122906714.png)

```sql
WITH p AS
  (SELECT SUM(CASE
                  WHEN reaction = 'like' THEN 1
                  ELSE 0
              END)/COUNT(*)::decimal AS prop,
          friend
   FROM facebook_reactions
   GROUP BY 2)
SELECT date_day,
       poster,
       avg(prop)
FROM facebook_reactions f
JOIN p ON f.friend= p.friend
GROUP BY 1,
         2
ORDER BY 3 DESC;
```

- The first concept here is the CTE which is initiated using the keyword WITH. It allocates the value 1 to every like, then sums the likes, and divides the sum by the number of all reactions. Data is then grouped by a friend. This is how we get the propensity defined in the question.
- The following SELECT statement uses the CTE to calculate the average propensity using the AVG() functions. To do that, you need to JOIN two tables, group data by date and poster, and order it by propensity in descending order.