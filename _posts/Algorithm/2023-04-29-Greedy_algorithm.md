---
title:  "Greedy Algorithm"
categories: algorithm
tag: [Data_Structure, python]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

The algorithm never reverses the earlier decision, even if the choice is wrong. It works in a top-down approach.

This algorithm may not produce the best result for all the problems. It's because it always goes for the best local choice to produce the best global result.

## Advantages and Drawback

- **Pros:** The algorithm is **easier to describe**.
- **Pros:** This algorithm can **perform better** than other algorithms (but not in all cases).
- **Cons:** The greedy algorithm doesn't always produce the optimal solution. This is the major disadvantage of the algorithm.

![image-20230429151051076](/../images/2023-04-29-Greedy_algorithm/image-20230429151051076.png)

Let's start with the root node **20**. Our problem is to find the largest path. And the optimal solution at the moment is **3**. Finally, the weight of an only child of **3** is **1**. This gives us our final result `20 + 3 + 1 = 24`.

## Example-1

```python
n = 1260
count = 0

array = [500, 100, 50, 10] #큰 단위의 화폐부터
coin = 500
for coin in array:
    count += n // coin
    n %= coin

print(count) 
#500 * 2 + 100 * 2 + 50 * 1 + 10 * 1
```





