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

| Methods    | Description                                                  |
| :--------- | :----------------------------------------------------------- |
| del a[0]   | **Remove items by index or slice**<br />l = [0, 10, 20, 30, 40, 50]  del l[0]  # [10, 20, 30, 40, 50] |
| a.remove() | **Remove an item by value**<br />l = ['Alice', 'Bob', 'Charlie', 'Bob', 'Dave']  l.remove('Alice') <br /># ['Bob', 'Charlie', 'Bob', 'Dave'] |
| a.pop()    | **Remove an item by index and get its value**<br />a = [0, 10, 20, 30, 40, 50] <br />print(l.pop(0)) #0 # [10, 20, 30, 40, 50] |
| a.clear()  | **Remove all items**                                         |

#### Add an item

| Methods                       | Description                                                  |
| :---------------------------- | :----------------------------------------------------------- |
| a.append('abc')               | Add an item to a list<br />l.append([3, 4, 5]), # [0, 1, 2, 100, [3, 4, 5]] |
| a.extend([10, 11]), `+`, `+=` | Combine lists<br />a = [0, 1, 2], # [0, 1, 2, 10, 11]<br />a +=[10, 11] # [0, 1, 2, 10, 11] |
| a.insert(1, 100)              | Insert an item into a list<br />a= ['a', 'b', 'c']  a.insert(1, 100) # ['a', 100, 'b', 'c'] |

#### The others

| Methods               | Description                                                  |
| :-------------------- | :----------------------------------------------------------- |
| a.sort(reverse=False) | *Sort* the *list* ascending by default.                      |
| a.reverse()           | Reverses the sorting order of the elements.                  |
| a.index(3)            | Returns the position at the first occurrence of the specified value. a = [1, 2, 3] #2 |
| a.count(1)            | a = [1, 2, 3, 1] #2                                          |
|                       |                                                              |

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