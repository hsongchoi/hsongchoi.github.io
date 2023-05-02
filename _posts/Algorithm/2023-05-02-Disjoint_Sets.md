---
title:  "Disjoint Sets' Algorithm"
categories: algorithm
tag: [Data_Structure, Python]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---



### 그래프

- 그래프(Graph)란 노드와 노드 사이에 연결된 간선의 정보를 가지고 있는 자료구조를 의미한다.

### 서로소 집합

- 서로소 집합(Disjoint Sets)이란 공통 원소가 없는 두 집합을 의미한다.

### 서로소 집합 자료구조

- 서로소 부분 집합들로 나누어진 원소들의 데이터를 처리하기 위한 자료구조이다.
- 서로소 집합 자료구조는 두 종류의 연산을 지원한다.
  `합집합(Union): 두 개의 원소가 포함된 집합을 하나의 집합으로 합치는 연산이다.`
  `찾기(Find): 특정한 원소가 속한 집합이 어떤 집합인지 알려주는 연산이다.`
- 서로소 집합 자료구조는 합치기 찾기(Union Find) 자료구조라고 불리기도 한다.
- 여러 개의 합치기 연산이 주어졌을 때 서로소 집합 자료구조의 동작 과정은 다음과 같습니다.
  ![img](https://velog.velcdn.com/images/yeahxne/post/1eb54c6f-6bd9-4e80-9553-965f6488a9b8/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/4be2408f-d9d5-4a25-adf6-c3eddaeb4933/image.png)

```
이러한 4개의 union 연산은 각각 '1과 4는 같은 집합','2와 3은 같은 집합','2와 4는 같은 집합','5와 6은 같은 집합'이라는 의미를 가지고 있다.
```

![img](https://velog.velcdn.com/images/yeahxne/post/4d2e0d2c-a299-4609-97e2-4261ce71a732/image.png)
`일반적으로 합치기 연산을 수행할 때, 더 큰 루트 노드가 더 작은 노드를 가리키도록 만들어서 테이블을 갱신한다.`

![img](https://velog.velcdn.com/images/yeahxne/post/50c82170-ec70-42dc-908c-4e65859305bf/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/6bc058cd-246c-4729-8497-034e0ba0e9f8/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/2ac938ee-f924-47ce-b170-e20e0a5805ed/image.png)

#### 서로소 집합 자료구조: 연결성

![img](https://velog.velcdn.com/images/yeahxne/post/77b81e6f-98b8-443b-bb23-668c5847a0b6/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/64f91c5a-5b0a-455f-83f9-e46962b51e2b/image.png)

#### 서로소 집합 자료구조: 기본적인 구현 방법

```python
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
	# 루트 노드를 찾을 때까지 재귀 호출
    if parent[x] != x:
    	return find_parent(parent, parent[x])
    return x

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
	a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a<b:
    	parent[b] = a
    else:
    	parent[a] = b
        
# 노드의 개수와 간선(Union 연산)의 개수 입력 받기
v,e = map(int, input().split())
parent = [0]*(v+1)  # 부모 테이블 초기화하기

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v+1):
	parent[i] = i
    
# Union 연산을 각각 수행
for i in range(e):
	a,b = map(int, input().split())
    union_parent(parent, a, b)
    
# 각 원소가 속한 집합 출력하기
print('각 원소가 속한 집합: ', end='')
for i in range(1, v+1):
	print(find_parent(parent, i), end=' ')
    
print()

# 부모 테이블 내용 출력하기
print('부모 테이블: ', end='')
for i in range(1, v+1):
	print(parent[i], end=' ')
```

#### 서로소 집합 자료구조: 기본적인 구현 방법의 문제점

![img](https://velog.velcdn.com/images/yeahxne/post/2f9e44de-1b4d-41bb-b29d-13d2926b16c3/image.png)

#### 서로소 집합 자료구조: 경로 압축

- 찾기(Find) 함수를 최적화하기 위한 방법으로 경로 압축(Path Compression)을 이용할 수 있다.
  `찾기(Find) 함수를 재귀적으로 호출한 뒤에 부모 테이블 값을 바로 갱신한다.`

```python
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
	# 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
    if paret[x] != x:
    	parent[x] = find_parent(parent, parent[x])
    return parent[x]
```

- 경로 압축 기법을 적용하면 각 노드에 대하여 찾기(Find) 함수를 호출한 이후에 해당 노드의 루트 노드가 바로 부모 노드가 된다.

```python
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
	# 루트 노드를 찾을 때까지 재귀 호출
    if parent[x] != x:
    	return find_parent(parent, parent[x])
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
	a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a<b:
    	parent[b] = a
    else:
    	parent[a] = b
        
# 노드의 개수와 간선(Union 연산)의 개수 입력 받기
v,e = map(int, input().split())
parent = [0]*(v+1)  # 부모 테이블 초기화하기

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v+1):
	parent[i] = i
    
# Union 연산을 각각 수행
for i in range(e):
	a,b = map(int, input().split())
    union_parent(parent, a, b)
    
# 각 원소가 속한 집합 출력하기
print('각 원소가 속한 집합: ', end='')
for i in range(1, v+1):
	print(find_parent(parent, i), end=' ')
    
print()

# 부모 테이블 내용 출력하기
print('부모 테이블: ', end='')
for i in range(1, v+1):
	print(parent[i], end=' ')
```

### 서로소 집합을 활용한 사이클 판별

- 서로소 집합은 무방향 그래프 내에서의 사이클을 판별할 때 사용할 수 있다.
  `참고로 방향 그래프에서의 사이클 여부는 DFS를 이용하여 판별할 수 있다.`
- 사이클 판별 알고리즘은 다음과 같다.
  ![img](https://velog.velcdn.com/images/yeahxne/post/5f091d49-fe73-4ac4-88b9-4d217ae6c4ea/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/de46f87a-ef85-40b7-b2d7-f8585a492add/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/8bbbcc50-c2d1-4444-a3c7-982c192b0fe8/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/34cd45cb-c509-44e3-bd0f-85c94f16ad06/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/f37adde8-d7f9-441e-8ece-2f9aaca4bf65/image.png)

```python
# 특정 원소가 속한 집합을 찾기
def find_parent(parent, x):
	# 루트 노드를 찾을 때까지 재귀 호출
    if parent[x] != x:
    	parent[x] = find_parent(parent, parent[x])
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union_parent(parent, a, b):
	a = find_parent(parent, a)
    b = find_parent(parent, b)
    if a<b:
    	parent[b] = a
    else:
    	parent[a] = b
        
# 노드의 개수와 간선(Union 연산)의 개수 입력 받기
v, e = map(int, input().split())
parent = [0]*(v+1)  # 부모 테이블 초기화하기

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v+1):
	parent[i] = i

cycle = False  # 사이클 발생 여부

for i in range(e):
	a,b = map(int, input().split())
    # 사이클이 발생한 경우 종료
    if find_parent(parent, a) == find_parent(parent, b):
    	cycle = True
        break
    # 사이클이 발생하지 않았다면 합집합(Union) 연산 수행
    else:
    	union_parent(parent, a, b)
        
if cycle:
	print("사이클이 발생했습니다.")
else:
	print("사이클이 발생하지 않았습니다.")
```