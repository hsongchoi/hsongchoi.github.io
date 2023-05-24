---
title:  "Python_Basic"
categories: algorithm
tag: [Data_types, python]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

# Data Types

## Numbers

- If we use 1e9, it will be `Floats`. Thus, we convert it into an integer with the function `int(1e9)`.
- If we produce the result in float, we recommend using the function `round()`.
  - Round(0.789, 2) = 0.79
- `2**3`: $2^3$, The exponentiation operator (**)
- `7%2`: 1, The modulo operator (%) returns the remainder of dividing two numbers.
- `7//2` : 3, The Floor division operator (//)

## Strings

- Multiline string: Use triple double quotes `"""` or triple single quotes `'''`

```python
a = '''
"Life" is short
you need python
'''
print(a)

#Output: Life is "short"
#        you need python
```

- Escape `\n`: **a newline** (In Python strings, the backslash \ is a special character, also called the "escape" character. )
- Join Two or More Strings

``` python
greet = "Hello, "
name = "Jack"

result = greet + name
print(result)
# Output: Hello, Jack
```

- Multiply a string

```python
print("=" * 10)
print("My Program")
print("=" * 10)
# Output: 
#==========
#My Program
#==========
```

- Braces: In order to make a brace appear in your string, you must use double braces {{ }}.

### Functions

| Methods                                                      | Description                                                  |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| ['abc'.upper()](https://www.programiz.com/python-programming/methods/string/upper) | converts the string to uppercase                             |
| ['ABC'.lower()](https://www.programiz.com/python-programming/methods/string/lower) | converts the string to lowercase                             |
| 'hobby'.count('b')                                           | Counts the specific character                                |
| **'hobby'.find('b')**                                        | returns the index of the first occurrence of a substring. If there does not exist a substring, it returns '-1'. |
| **'hobby'.index('b')**                                       | returns the index of the first occurrence of a substring. If there does not exist a substring, it returns an error message. |
| "    hi.     ".rstrip()                                      | removes right spaces.                                        |
| "    hi.     ".lstrip()                                      | removes left spaces.                                         |
| "    hi.     ".strip()                                       | removes all spaces.                                          |
| d.split(), e.split(':')                                      | splits string from left, d = "Life is too short", e = "ab:c:d" |
| ",".join('abcd')                                             | #'a,b,c,d'                                                   |
| c.replace("Life", "Your leg")                                | c = "Life is too short."                                     |

### String Formatting (f-Strings)

Python **f-Strings** make it really easy to print values and variables. The `f` in f-strings may as well stand for “fast.” f-strings are faster than both %-formatting and `str.format()`.

```python
name = 'Hailey'
country = 'the States'

print(f'{name} is from {country}')
#Hailey is from the States

# Dictionary
d = {'name': 'Hailey', 'country':'the States'}
print(f'{d["name"]} is from {d["country"]}')

#
```

If you want to spread strings over multiple lines, you also have the option of escaping a return with a `\`:

```python
message = f"Hi {name}. " \
...       f"You are a {profession}. " \
...       f"You were in {affiliation}."
...
#message
#'Hi Eric. You are a comedian. You were in Monty Python.'
```

| Methods        | Description  |
| :------------- | :----------- |
| f'{"hi":<10}'  | 'hi        ' |
| f'{"hi":>10}'  | '        hi' |
| f'{"hi":^10}'  | '    hi    ' |
| f'{"hi":=^10}' | '====hi====' |
| f'{"hi":!<10}' | 'hi!!!!!!!!' |
| f'{y:0.4f}'    | '3.4213'     |
| f'{y:10.4f}'   | '    3.4213' |

## Lists

### Functions

#### Remove items

| Methods          | Description                                                  |
| :--------------- | :----------------------------------------------------------- |
| del a[0]         | **Remove items by index or slice**<br />l = [0, 10, 20, 30, 40, 50]  del l[0]  # [10, 20, 30, 40, 50] |
| a.remove(): O(N) | **Remove an item by value**<br />l = ['Alice', 'Bob', 'Charlie', 'Bob', 'Dave']  l.remove('Alice') <br /># ['Bob', 'Charlie', 'Bob', 'Dave'] |
| a.pop()          | **Remove an item by index and get its value**<br />a = [0, 10, 20, 30, 40, 50] <br />print(l.pop(0)) #0 # [10, 20, 30, 40, 50] |
| a.clear()        | **Remove all items**                                         |

#### Add an item

| Methods                       | Description                                                  |
| :---------------------------- | :----------------------------------------------------------- |
| a.append('abc') : O(1)        | Add an item to a list<br />l.append([3, 4, 5]), # [0, 1, 2, 100, [3, 4, 5]] |
| a.extend([10, 11]), `+`, `+=` | Combine lists<br />a = [0, 1, 2], # [0, 1, 2, 10, 11]<br />a +=[10, 11] # [0, 1, 2, 10, 11] |
| a.insert(1, 100): O(N)        | Insert an item into a list<br />a= ['a', 'b', 'c']  a.insert(1, 100) # ['a', 100, 'b', 'c'] |

#### The others

| Methods                         | Description                                                  |
| :------------------------------ | :----------------------------------------------------------- |
| a.sort(reverse=False): O(NlogN) | *Sort* the *list* ascending by default.                      |
| a.reverse(): O(N)               | Reverses the sorting order of the elements.                  |
| a.index(3)                      | Returns the position at the first occurrence of the specified value. a = [1, 2, 3] #2 |
| a.count(1): O(N)                | a = [1, 2, 3, 1] #2                                          |
|                                 |                                                              |

### List Comprehension

> array = [i for i in A if ____]

```python
array = [i for i in range(4)]
print(array) #[0, 1, 2, 3]

fruits = ["apple", "banana", "cherry", "mango"]
newlist = [i for i in fruits if "a" in i]
print(newlist) #['apple', 'banana', 'mango']

a=[1,2,3,4]
result = [num * 3 for num in a if num %2 == 0] #[6, 12]
print(result)
```

#### Initialize a list with the given size

```python
# N * M
n = 4
m = 3
array = [ [0]*m for _ in range(n) ] # Not [ [0]*m ] * n 
#Underscore : as a variable in looping
```

#### How to zip two lists 

##### 1. **Using the built-in zip() function**

```python
list_a = [1, 3, 4]
list_b = [5, 7, 11]

list_zip = list(zip(list_a, list_b))
print(list_zip) 
#[(1, 5), (3, 7), (4, 11)]
```

##### 2. **Using map() + __add__**

map() function is another **in-built python method** similar to zip() function above. It enables you to zip the elements of the iterable by **mapping the elements of the first iterable with the elements of the second iterable**. By using the map() function along with the addition operator, you can merge two lists in python as shown in the below example:

```python
list_1 = [[2, 3], [4, 5], [7, 6]]
list_2 = [[4, 9], [4, 2], [11, 10]]
  
print ("The given list 1 is : " + str(list_1))
print ("The given list 2 is : " + str(list_2))
#The given list 1 is : [[2, 3], [4, 5], [7, 6]]
#The given list 2 is : [[4, 9], [4, 2], [11, 10]]


res = list(map(list.__add__, list_1, list_2))
      
print ("The zipped list is : " +  str(res))

#The zipped list is : [[2, 3, 4, 9], [4, 5, 4, 2], [7, 6, 11, 10]]
```

## Tuples

- Tuples that contain `immutable elements` can be used as a key for a dictionary. With lists, this is not possible.
- If you have data that doesn't change, implementing it as a tuple will guarantee that it remains write-protected.

```python
# Empty Tuples
my_tuple = ()

# nested tuple
my_tuple = ("mouse", [8, 4, 6], (1, 2, 3))

#We can also create tuples without using parentheses:
my_tuple = 1, "Hello", 3.4
```

#### Creating a Tuple with one Element

In Python, creating a tuple with one element is a bit tricky. Having one element within parentheses is not enough.

```python
var1 = ("hello")
print(type(var1))  # <class 'str'>

# Creating a tuple having one element
var2 = ("hello",)
print(type(var2))  # <class 'tuple'>

# Parentheses is optional
var3 = "hello",
print(type(var3))  # <class 'tuple'>
```

#### Check if an Item Exists

```python
languages = ('Python', 'Swift', 'C++')

print('C' in languages)    # False
print('Python' in languages)    # True
```

## Dictionaries

```python
# Empty Dictionaries
my_dictionary = {}

#Add Elements to a Python Dictionary
capital_city = {"Nepal": "Kathmandu", "England": "London"}
capital_city["Japan"] = "Tokyo"
print("Updated Dictionary: ",capital_city)

#Removing elements
student_id = {111: "Eric", 112: "Kyle", 113: "Butters"}
print("Initial Dictionary: ", student_id)
del student_id[111]

```

| Methods    | Description                                                  |
| :--------- | :----------------------------------------------------------- |
| a.keys()   | Returns the list of keys.                                    |
| a.values() | Returns the list of values.                                  |
| a.items()  | Returns the key-value pairs of the dictionary as tuples in a list. |
| a.get(key) | Returns the value of the "key"                               |

## Sets

```python
# Create an empty set
empty_set = set()

# Create an empty dictionary
empty_dictionary = { }

s1 = set([1, 2, 3, 4, 5, 6])
s2 = set([4, 5, 6, 7, 8, 9])
print(s1 & s2)
print(s1 | s2)
print(s1 - s2)

```

| Methods                  | Description                                                  |
| :----------------------- | :----------------------------------------------------------- |
| a.add(32)                | Add an item to a set<br />Initial Set: {34, 12, 21, 54} Updated Set: {**32**, 34, 12, 21, 54} |
| a.update(tech_companies) | Update the set with items of other collection types (lists, tuples, sets, etc).<br />companies = {'Lacoste', 'Ralph Lauren'} tech_companies = ['apple', 'google', 'apple']<br /># Output: {'google', 'apple', 'Lacoste', 'Ralph Lauren'} |
| a.items()                | Returns the key-value pairs of the dictionary as tuples in a list. |

# Input

## How to take integer input

```python
n = int(input())

data = list(map(int, input().split())) #From string to integer

import sys
data = sys.stdin.readline().rstrip()
print(data)

#used for the addition of any string at the end of the output of the python print statement.
print(7, end = " ")
```

# Conditions

## Pass statement

When the `pass` statement is executed, nothing happens, but you avoid getting an error when empty code is not allowed.

```python
def myfunction():
  pass

class Person:
  pass

if b > a:
  pass
```

## Conditional Expression

```python
score = 85
result = "Success" if score >= 80 else "Fail" #Success
```

# For & While Loops

```python
# While loop
i = 1
result = 0

while i <= 9:
  if i % 2 == 1:
    result += i
  i += 1
  
print(result)

# For loop
result = 0
for i in range(1, 10):
  result += i
  print(result) #Sum of 1- 9: 45
  
```

## Break statement in For Loop

Breakpoint is a unique function in For Loop that allows you to break or terminate the execution of the for loop. We declared the numbers from 10-20, but we want that our for loop to terminate at number 15 and stop executing further.

```python
for x in range (10,20):
			if (x == 15): break
			print(x)
```

## Continue statement in For Loop

`Continue function`, as the name indicates, will `terminate the current iteration` of the for loop `BUT will continue execution of the remaining iterations`. In our example, we have declared values 1-9, but between these numbers, we only want those numbers that are NOT  divisible by 2.

```python
#Sum of odd integers from 1 to 9
result = 0
for i in range(1, 10):
  if i % 2 == 0: continue
  result += i
print(result) # 25
```

## Enumerate() in For Loop

**enumerate()** is a built-in function used for assigning an index to each item of the iterable object. It adds a loop on the iterable objects while keeping track of the current item and returns the object in an enumerable form. This object can be used in a for loop to convert it into a list by using list() method.

```python
#use a for loop over a collection
Months = ["Jan","Feb","Mar","April","May","June"]
for i, m in enumerate (Months):
		print(i,m)
    
0 Jan
1 Feb
2 Mar
3 April
4 May
5 June
```

# Function

## Global variables

When you create a variable inside a function, that variable is local and can only be used inside that function. If you use the `global` keyword, the variable belongs to the global scope:

```python
def myfunc():
  global x
  x = "fantastic"

myfunc()
print("Python is " + x) # Python is fantastic
```

## Multiple returns

As you already know, a function can return a single variable, but it can also return multiple variables.

```python
def getPerson():
    name = "Leona"
    age = 35
    country = "UK"
    return name,age,country

name,age,country = getPerson()
```

## Lambda function 

In Python, a lambda function is a special type of function without the function name. 

> ```python
> lambda argument(s) : expression 
> 
> greet = lambda : print('Hello World')
> greet() #Call the lambda
> ```

- `argument(s)` - any value passed to the lambda function
- `expression` - expression is executed and returned

```python
print( (lambda a, b: a + b)(3, 7) ) #10
```

# Built-in Functions

### **sorted**()

sorted( [9, 1, 8, 5, 4], *reverse=True*) : Return a new sorted list from the items in *iterable*. #[9, 8, 5, 4, 1]

```python
array = [('Hailey', 34), ('James', 30), ('Mike', 29)]
result = sorted(array, key = lambda x: x[1], reverse=True)
print(result)
```

### **counter()**

```python
from collections import Counter
counter = Counter(['red', 'blue', 'red', 'green', 'blue', 'blue'])
print(counter['blue'])
print(counter['green'])
print(dict(counter))
```



### **Permutations** and **Combination**

```python
from itertools import permutations
data = ['A','B','C']
result = list(permutations(data, 3))
print(result)

from itertools import combinations
result = list(combinations(data, 2))
print(result)

from itertools import combinations_with_replacement
result = list(combinations_with_replacement(data, 2))
print(result)
```

### **GCD** and **LCM**

```python
import math

def lcm(a, b)
		return a * b // math.gcd(a, b)

a = 21
b = 14
print(math.gcd(21, 14))
print(math.lcm (21, 14))
```

