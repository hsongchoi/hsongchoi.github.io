---
title:  "Optimization Algorithm"
categories: ML
tag: [Python, Machine-Learning]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

<head>
  <style>
    table.dataframe {
      white-space: normal;
      width: 100%;
      height: 240px;
      display: block;
      overflow: auto;
      font-family: Arial, sans-serif;
      font-size: 0.9rem;
      line-height: 20px;
      text-align: center;
      border: 0px !important;
    }

    table.dataframe th {
      text-align: center;
      font-weight: bold;
      padding: 8px;
    }

    table.dataframe td {
      text-align: center;
      padding: 8px;
    }

    table.dataframe tr:hover {
      background: #b8d1f3; 
    }

    .output_prompt {
      overflow: auto;
      font-size: 0.9rem;
      line-height: 1.45;
      border-radius: 0.3rem;
      -webkit-overflow-scrolling: touch;
      padding: 0.8rem;
      margin-top: 0;
      margin-bottom: 15px;
      font: 1rem Consolas, "Liberation Mono", Menlo, Courier, monospace;
      color: $code-text-color;
      border: solid 1px $border-color;
      border-radius: 0.3rem;
      word-break: normal;
      white-space: pre;
    }

  .dataframe tbody tr th:only-of-type {
      vertical-align: middle;
  }

  .dataframe tbody tr th {
      vertical-align: top;
  }

  .dataframe thead th {
      text-align: center !important;
      padding: 8px;
  }

  .page__content p {
      margin: 0 0 0px !important;
  }

  .page__content p > strong {
    font-size: 0.8rem !important;
  }

  </style>
</head>


## Gradient Descent (GD)


`Gradient descent (GD)` is an iterative first-order optimization algorithm used to minimize the cost function.



- The function has to be differentiable and convex.

    - Typical non-differentiable functions have a cusp or a discontinuity.

    - <img src="https://miro.medium.com/v2/resize:fit:686/1*JS6lMwXqsos9XfDxIwLVPQ.gif"> where 0 < lambda <1. 

- `Gradient Descent Algorithm` iteratively calculates the next point using gradient at the current position, scales it (by a `learning rate`) and subtracts obtained value from the current position (makes a step).

- We implement this formula by `taking the derivative (the tangential line to a function) of our cost function`. The `slope` of the `tangent line` is the value of the derivative at that point and it will give us a `direction to move towards`. We make steps down the cost function in the direction with the steepest descent.

- <img src ="https://images.velog.io/images/arittung/post/c042f965-cbc2-4ab7-9e60-3194ed2038b0/image.png">

where eta controls the step size.

- The smaller learning rate the longer GD converges, or may reach maximum iteration before reaching the optimum point. On the other hand, if learning rate is too big the algorithm may not converge to the optimal point (jump around) or even to diverge completely.



This function takes 5 parameters:



1. `starting point` - in our case, we define it manually but in practice, it is often a random initialisation



2. `gradient function` - has to be specified before-hand



3. `learning rate` - scaling factor for step sizes



4. `maximum number of iterations`



5. `tolerance` to conditionally stop the algorithm (in this case a default value is 0.01)


### Learning rate



It determines the size of the steps that are taken by the gradient descent algorithm.



If the `learning rate is too big`, the loss will bounce around and may not reach the local minimum.



If the `learning rate is too small` then gradient descent will eventually reach the local minimum but require a long time to do so.



`The cost function should decrease` over time if gradient descent is working properly.


### Pros and cons



- A simple algorithm that is easy to implement and each iteration is cheap; we just need to compute a gradient

- However, it’s often slow because many interesting problems are not strongly convex

- Cannot handle non-differentiable functions (biggest downside)




```python
import numpy as np

def gradient_descent(start, gradient, learn_rate, max_iter, tol=0.01):
  steps = [start] # history tracking
  x = start

  for _ in range(max_iter):
    diff = learn_rate*gradient(x)
    if np.abs(diff)<tol:
      break    
    x = x - diff
    steps.append(x) # history tracing

  return steps, x
```

### A quadratic function



```python
def func1(x):
  return x**2-4*x+1

def gradient_func1(x):
  return 2*x - 4

history, result = gradient_descent(9, gradient_func1, 0.1, 100)
```

<img src ="https://miro.medium.com/v2/resize:fit:1400/1*v5bc1TzeMKpzTAgorjOoHQ.gif">


### A function with a saddle point


`Disadvantage`: The problem with gradient descent is that converging to a local minimum takes `extensive time` and determining a `global minimum is not guaranteed`.


The existence of a saddle point imposes a real challenge for the first-order gradient descent algorithms like GD and obtaining a global minimum is not guaranteed. Second-order algorithms deal with these situations better (e.g. Newton-Raphson method).


<img src ="https://miro.medium.com/v2/resize:fit:1080/1*Z3wjHCFAehYXYG9lOiqiEw.gif">


## Stochastic Gradient Descent 


In SGD, we use `one training sample at each iteration` instead of using the whole dataset to sum all for every step, that is — SGD performs a parameter update for each observation. 



- So instead of looping over each observation, it just needs one to perform the parameter update.

- In SGD, before for-looping, we need to `randomly shuffle` the training examples.

- SGD is usually faster than batch gradient descent, but its path to the minima is `more random` than batch gradient descent since SGD uses `only one example` at a time.

- SGD is widely used for `larger dataset training`, computationally faster, and can allow for parallel model training because it has to have a memory for one sample.



<img src ="https://images.velog.io/images/arittung/post/72ed8c88-c35b-4535-8e7a-b0acac4144be/image.png">


## Mini-Batch Gradient Descent


- Splitting the training data into smaller batches.

- Rather than calculating the gradient of a single training example, the gradient is calculated against a batch size of n training examples (Mini-batch gradient descent uses n data points (instead of one sample in SGD) at each iteration.).

- `Samples in each gradient calculation`: consecutive subsets of the dataset


### Batch size



- `mini Batch`: equally sized subsets of the dataset over which the gradient is calculated and weights updated (A subset of all your data during one iteration).



### Epoch

- One loop over the entire training dataset.

- Too many epochs can lead to overfitting of the training dataset, whereas too few may result in an underfit model, but decrease the loss.

- `Early stopping` is a method that allows you to specify an arbitrarily large number of training epochs and stop training once the model performance stops improving on the validation dataset.



<img src ="https://images.velog.io/images/arittung/post/a92158ca-80f0-46dc-97a2-22006e1b13c8/image.png">



```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

np.random.seed(0)

X = 2 * np.random.rand(100,1)
y = 6 +4 * X+ np.random.randn(100,1)

plt.scatter(X, y)
```

<pre>
<matplotlib.collections.PathCollection at 0x7fc1c881f0a0>
</pre>
<img src="data:image/png;base64,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"/>


```python
def stochastic_gradient_descent_steps(X, y, batch_size=10, iters=1000):
    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    prev_cost = 100000
    iter_index =0
    
    for ind in range(iters):
        np.random.seed(ind)
        #sample_X, sample_y
        stochastic_random_index = np.random.permutation(X.shape[0])
        sample_X = X[stochastic_random_index[0:batch_size]]
        sample_y = y[stochastic_random_index[0:batch_size]]
        #w1_update, w0_update
        w1_update, w0_update = get_weight_updates(w1, w0, sample_X, sample_y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
    
    return w1, w0
```


```python
def get_weight_updates(w1, w0, X, y, learning_rate=0.01):
    N = len(y)

    w1_update = np.zeros_like(w1)
    w0_update = np.zeros_like(w0)

    y_pred = np.dot(X, w1.T) + w0
    diff = y-y_pred
         
    w0_factors = np.ones((N,1))

    w1_update = -(2/N)*learning_rate*(np.dot(X.T, diff))
    w0_update = -(2/N)*learning_rate*(np.dot(w0_factors.T, diff))    
    
    return w1_update, w0_update
```


```python
def gradient_descent_steps(X, y, iters=10000):

    w0 = np.zeros((1,1))
    w1 = np.zeros((1,1))
    
    for ind in range(iters):
        w1_update, w0_update = get_weight_updates(w1, w0, X, y, learning_rate=0.01)
        w1 = w1 - w1_update
        w0 = w0 - w0_update
              
    return w1, w0
```


```python
def get_cost(y, y_pred):
    N = len(y) 
    cost = np.sum(np.square(y - y_pred))/N
    return cost
```


```python
w1, w0 = stochastic_gradient_descent_steps(X, y, iters=1000)
print("w1:",round(w1[0,0],3),"w0:",round(w0[0,0],3))
y_pred = w1[0,0] * X + w0
print('Stochastic Gradient Descent Total Cost:{0:.4f}'.format(get_cost(y, y_pred)))
```

<pre>
w1: 4.028 w0: 6.156
Stochastic Gradient Descent Total Cost:0.9937
</pre>
## Newton-Raphson


It can be easily generalized to the problem of finding solutions of a system of non-linear equations, which is referred to as Newton's technique. 



- It requires only one inital value $x_0$, which we will refer to as the initial guess for the root.

- To see how the N-R method works, we can rewrite the function f(x) using a `Taylor series expansion` in (x-x0):

$f(x) = f(x_0) + f'(x_0)(x-x_0) + 1/2 f''(x_0)(x-x_0)^2 + ... = 0 $

where f'(x) denotes the first derivative of f(x) with respect to x, f''(x) is the second derivative, and so forth. 

- Now, suppose the initial guess is pretty close to the real root. Then (x-x0) is small, and only the first few terms in the series are important to get an accurate estimate of the true root, given x0.

- By truncating the series at the second term (linear in x), we obtain the N-R iteration formula for getting a better estimate of the true root:

<img src =" https://web.mit.edu/10.001/Web/Course_Notes/NLAE/equation6.gif">

- Thus the N-R method finds the tangent to the function f(x) at $x=x_0$ and extrapolates it to intersect the x axis to get $x_1$. This point of intersection is taken as the new approximation to the root and the procedure is repeated until convergence is obtained whenever possible. Mathematically, given the value of $x = x_i$ at the end of the ith iteration, we obtain $x_{i+1}$ as


### Pros and cons



`Pros`:

- It is best method to solve the non-linear equations.

- The order of convergence is quadric i.e. of second order which makes this method fast as compared to other methods.

- It is very easy to implement on computer.



`Cons`:

- This method becomes complicated `if the derivative of the function f(x) is not simple`.

- This method requires a great and sensitive attention regarding the choice of its approximation.



```python
def func(x):
    return x * x * x - x * x + 2
 
# Derivative of the above function
# which is 3*x^x - 2*x
def derivFunc(x):
    return 3 * x * x - 2 * x
 
# Function to find the root
def newtonRaphson(x):
    h = func(x) / derivFunc(x)
    while abs(h) >= 0.0001:
        h = func(x)/derivFunc(x)
         
        # x(i+1) = x(i) - f(x) / f'(x)
        x = x - h
     
    print("The value of the root is : ",
                             "%.4f"% x)
 
# Driver program to test above
x0 = -20 # Initial values assumed
newtonRaphson(x0)
 
```

<pre>
The value of the root is :  -1.0000
</pre>