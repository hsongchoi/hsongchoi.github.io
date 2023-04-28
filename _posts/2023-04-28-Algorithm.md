# Algorithm

## Complexity

Algorithmic **complexity** is concerned with how fast or slow a particular algorithm performs. We define complexity as a numerical function *T(n)* - time versus the input size *n*. 

### Time complexity

: The `time complexity` of an algorithm quantifies the amount of time taken by an algorithm to run as a function of the length of the input. 

#### Definition of Big-O

For any monotonic functions $f(n)$ and $g(n)$ from the positive integers to the positive integers, we say that $f(n) = O(g(n))$ when there exist constants $c > 0$ and $n_0 > 0$ such that
$$
f(n) \leq c * g(n)~for~all~n \geq n_0
$$

> The function f(n) **does not grow faster than g(n)**, or that function g(n) is an **upper bound** for f(n) for all sufficiently large $n→\infty$.

Here is a graphic representation of $f(n) = O(g(n))$ relation:

<img src="../images/2023-04-28-Algorithm/image-20230428104612112.png" alt="image-20230428104612112" style="zoom: 67%;" />