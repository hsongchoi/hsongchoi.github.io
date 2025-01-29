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

## Easy question 

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

# Basic Aggregate Functions



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

# GROUP BY Clause

## [3 Bed Minimum](https://platform.stratascratch.com/coding/9627-3-bed-minimum?python=&code_type=3)

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

## [Top 10 Songs](https://platform.stratascratch.com/coding/9743-top-10-songs?python=&code_type=3)

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

## [Find the top 5 cities with the most 5-star businesses](https://platform.stratascratch.com/coding/10148-find-the-top-10-cities-with-the-most-5-star-businesses?python=&code_type=3)

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
- Next, data is grouped by the customer’s first name and order date. You need to output the customer with the highest daily total order. To do that, you need the HAVING clause to get data where the sum of the order costs per customer and per date is equal to the order maximum. This is where you need another aggregate functions, which is MAX().

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