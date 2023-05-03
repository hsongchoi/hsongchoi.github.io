---
title:  "SQL: Data Manipulation"
categories: SQL
tag: [SQL, Data_Cleaning]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

# Add data to a table

## INSERT

: Adds a single row to the table

```sql
INSERT INTO MyTable (col1, col2)
VALUES ('value1', 'value2');
```

```sql
INSERT INTO people (first_name, last_name)
VALUES
('George', 'White'),
('Jenn', 'Smith'),
('Carol', NULL);
```

# Modify data in a table

## UPDATE

: Updates table data

```sql
UPDATE MyTable
SET col1 = 56
WHERE col2 = 'something';
```

-  WHERE clause: it targets only the correct record.
- We need to be as specific as possible with our statement.

```sql
UPDATE people
SET last_name = 'Morrison'
WHERE id_number = 175;
```

# Remove data from a table

## DELETE

: Removes rows from the table

```sql
DELETE FROM MyTable
WHERE col1 = 'something';
```

```sql
DELETE FROM people
WHERE id_number = 1001;
```

```sql
DELETE FROM people
WHERE quiz_points IS NULL;
```

