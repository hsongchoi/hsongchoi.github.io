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
ORDER BY sum_n_messages DESC;
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