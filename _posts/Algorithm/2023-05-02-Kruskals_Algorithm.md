---
title:  "Kruskal's Algorithm"
categories: algorithm
tag: [Data_Structure, Python]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

#### 신장 트리

- 그래프에서 모든 노드를 포함하면서 사이클이 존재하지 않는 부분 그래프를 의미한다.
  `- 모든 노드가 포함되어 서로 연결되면서 사이클이 존재하지 않는다는 조건은 트리의 조건이기도 하다.`
  ![img](https://velog.velcdn.com/images/yeahxne/post/96484eab-8ab3-409a-8753-497510ed2550/image.png)

#### 최소 신장 트리

- 최소한의 비용으로 구성되는 신장 트리를 찾아야 할 때 어떻게 해야 할까요?
- 예를 들어 N개의 도시가 존재하는 상황에서 두 도시 사이에 도로를 놓아 전체 도시가 서로 연결될 수 있게 도로를 설치하는 경우를 생각해 봅시다.
  `- 두 도시 A,B를 선택했을 때 A에서 B로 이동하는 경로가 반드시 존재하도록 도로를 설치합니다.`
  ![img](https://velog.velcdn.com/images/yeahxne/post/2dc23195-dc5b-4e43-a1c3-6b0e870f5d2f/image.png)

### 크루스칼 알고리즘

- 대표적인 최소 신장 트리 알고리즘이다.
- 그리디 알고리즘으로 분류된다.
- 구체적인 동작 과정은 다음과 같다.
  ![img](https://velog.velcdn.com/images/yeahxne/post/2735b6a7-1918-41f3-ada2-99bbd7b584b9/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/75afe273-f71a-48c9-b1bc-ec6731ddef9b/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/103f7f19-fd6b-4653-a8ff-3a8f80d30847/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/9a0d17b6-43ab-4cc1-9b91-46ad17f102fe/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/fd3b9c98-f3f7-421a-ab2b-f192c6ee5b55/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/2e286f86-25a1-40bf-90ee-c0458a3985ef/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/5eb2f5d1-2583-47c9-976c-4f99f403ad9e/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/51c59206-49e4-48a5-a562-84507483f3bc/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/4110cdc9-8c9e-4447-bd33-eee0e796c834/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/70cd8a2a-bb2b-4fb4-a94d-2da5890709b1/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/2eae5375-600f-4946-a59b-e40ba09dd32b/image.png)

### 크루스칼 알고리즘: 코드

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
v,e = map(int, input().split())
parent = [0]*(v+1)  # 부모 테이블 초기화하기

# 모든 간선을 담을 리스트와, 최종 비용을 담을 변수
edges = []
result = 0

# 부모 테이블상에서, 부모를 자기 자신으로 초기화
for i in range(1, v+1):
	parent[i] = i
    
# 모든 간선에 대한 정보를 입력 받기
for _ in range(e):
	a,b,cost = map(int, input().split())
    # 비용순으로 정렬하기 위해서 튜플의 첫 번째 원소를 비용으로 설정
    edges.append((cost, a, b))

# 간선을 비용순으로 정렬
edges.sort()

# 간선을 하나씩 확인하며
for edge in edges:
	cost, a, b = edge
    # 사이클이 발생하지 않는 경우에만 집합에 포함
    if find_parent(parent, a) != find_parent(parent, b):
    	union_parent(parent, a, b)
        result += cost

print(result)
```

### 크루스칼 알고리즘 성능 분석

- 크루스칼 알고리즘은 간선의 개수가 E개일 때, *O*(*E**l**o**g**E*)의 시간 복잡도를 가진다.
- 크루스칼 알고리즘에서 가장 많은 시간을 요구하는 곳은 간선에 대해서 정령을 수행하는 부분이다.
  `- 표준 라이브러리를 이용해 E개의 데이터를 정렬하기 위한 시간 복잡도는 O(ElogE)이다.`
