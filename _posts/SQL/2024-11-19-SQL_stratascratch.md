---
title: "SQL: Strata Scratch test"
categories: SQL
tag: [SQL, Data_Cleaning]
author_profile: false
typora-root-url: ../
search: true
use_math: true

---

# Q1. Top Cool Votes

![image-20241119182706852](/images/2024-11-19-SQL_stratascratch/image-20241119182706852.png)

![image-20241119182804544](/images/2024-11-19-SQL_stratascratch/image-20241119182804544.png)

```sql
select business_name, review_text
from yelp_reviews
where cool = 
    (select max(cool)
     From yelp_reviews);
```

