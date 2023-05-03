---
title:  "SQL: Joins"
categories: SQL
tag: [SQL, Data_Cleaning]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---


# 1. (Inner) JOIN

When we ask to join tables based on some given criteria, that creates an overlap between the tables where the criteria match. Inner and outer refer to the space where the records match or overlap.

![image-20230503062107251](/images/2023-05-03-Two_more_tables/image-20230503062107251.png)

- CA and VA match up, but DE and MA don’t match.
- CA and VA would be considered inner because they're in the overlap space, and DE and MA would be considered outer. 

```sql
SELECT *
FROM people
JOIN states ON people.state = states.abbr;
```

## Implicit Joint

- Without join. That works just the same as writing join explicitly.
- `It's usually best to use the explicit join syntax`, just to keep things clear. It's also possible to end up with unintended consequences like a cross join when using an implicit join, so be sure to use explicit join syntax to avoid that.

```sql
SELECT people.first_name, people.state_code, states.division
FROM people, states
WHERE people.state_code = states.state_abbrev;
```

# LEFT JOIN

![image-20230503062426726](/images/2023-05-03-Two_more_tables/image-20230503062426726.png)

```sql
SELECT *
FROM people, states
LEFT JOIN	states ON people.state_code = states.state_abbrev;
```

# RIGHT JOIN

```sql
SELECT *
FROM people, states
RIGHT JOIN	states ON people.state_code = states.state_abbrev;
```

# FULL OUTER JOIN

```sql
SELECT *
FROM people, states
FULL OUTER JOIN	states ON people.state_code = states.state_abbrev;
```

