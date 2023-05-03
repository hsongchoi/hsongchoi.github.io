---
title:  "Dijkstra's Shortest Path Algorithm"
categories: algorithm
tag: [Data_Structure, Python]
author_profile: false
typora-root-url: ../
search: true
use_math: true
---

Dijkstra's algorithm allows us to find the shortest path between any two vertices of a graph.

It differs from the minimum spanning tree because the shortest distance between two vertices might not include all the vertices of the graph.

- 최단 경로 알고리즘은 가장 짧은 경로를 찾는 알고리즘을 의미한다.

- 다양한 문제 상황

  - 한 지점에서 다른 한 지점까지의 최단 경로

  - 한 지점에서 다른 모든 지점까지의 최단 경로

  - 모든 지점에서 다른 모든 지점까지의 최단 경로

- 각 지점은 그래프에서 노드로 표현
- 지점 간 연결된 도로는 그래프에서 간선으로 표현
  ![img](https://velog.velcdn.com/images/yeahxne/post/eb697590-9346-460d-806a-967e6bc6f79d/image.png)

# How Dijkstra's Algorithm works

- **특정한 노드**에서 출발하여 **다른 모든 노드**로 가는 최단 경로를 계산한다.
- 다익스트라 최단 경로 알고리즘은 음의 간선이 없을 때 정상적으로 동작한다.
  - 현실 세계의 도로(간선)은 음의 간선으로 표현되지 않습니다.
- 다익스트라 최단 경로 알고리즘은 **그리디 알고리즘으로 분류**됩니다.
  - 매 상황에서 가장 비용이 적은 노드를 선택해 임의의 과정을 반복합니다.
- Dijkstra uses this property in the opposite direction, i.e., we overestimate the distance of each vertex from the starting vertex. Then `we visit each node and its neighbors to find the shortest subpath to those neighbors.` 
- The algorithm uses a `greedy approach` in the sense that we find the next best solution `hoping that the end result is the best solution for the whole problem.`

- 알고리즘의 **동작 과정**은 다음과 같습니다.

```null
1. 출발 노드를 설정합니다.
2. 최단 거리 테이블을 초기화합니다.
3. 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택합니다.
4. 해당 노드를 거쳐 다른 노드로 가는 비용을 계산하여 최단 거리 테이블을 갱신합니다.
5. 위 과정에서 3번과 4번을 반복합니다.
```

![img](https://velog.velcdn.com/images/yeahxne/post/84145513-e001-47a0-9220-911f61fedcfa/image.png)

## Dijkstra's Algorithm: 동작 과정 살펴보기

![img](https://velog.velcdn.com/images/yeahxne/post/73f28212-b412-4d82-87d7-c0ff395322ad/image.png)

```
- 출발 노드까지의 거리가 0이고, 다른 모든 노드에 대해서는 거리가 무한인 것으로 초기화할 수 있다.
```

![img](https://velog.velcdn.com/images/yeahxne/post/d93ca029-2f88-4f46-8d64-386c27fb24c4/image.png)

```
- 1번 노드에서 2번 노드로 거쳐갈 때, 현재 값인 무한보다 2가 더 작기 때문에 2로 갱신한다. → 최단 거리 값이 2로 바뀐 것을 확인할 수 있다.`
`- 마찬가지로 3번, 4번 노드에 대해서 동일한 로직으로 갱신한다.
```

![img](https://velog.velcdn.com/images/yeahxne/post/4f0f13ed-bd13-49a0-81f9-b97e051152f9/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/92ebdb33-28e7-483c-99f9-128647422c3c/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/143022b8-cdf1-4584-bfde-91e0683a55bb/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/09b5d86f-57f2-4fd9-afc6-1e0f69d58dfa/image.png)

## Dijkstra's Algorithm의 특징

- 그리디 알고리즘 (Greedy Algorithm): 매 상황에서 방문하지 않은 가장 비용이 적은 노드를 선택해 임의의 과정을 반복한다.
- 단계를 거치며 한 번 처리된 노드의 최단 거리는 고정되어 더이상 바뀌지 않습니다.
  `- 한 단계당 하나의 노드에 대한 최단 거리를 확실히 찾는 것으로 이해할 수 있습니다.`
- 다익스트라 알고리즘을 수행한 뒤에 테이블에 각 노드까지의 최단 거리 정보가 저장됩니다.
  `- 완벽한 형태의 최단 경로를 구하려면 소스코드에 추가적인 기능을 더 넣어야 합니다.`

# Code for Dijkstra's Algorithm: 간단한 구현

- 단계마다 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택하기 위해 매 단계마다 1차원 테이블의 모든 원소를 확인(순차 탐색)합니다.

```python
import sys
input = sys.stdin.readline
INF = int(1e9)  # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수를 입력받기
n,m = map(int, input().split())
# 시작 노드 번호를 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n+1)]
# 방문한 적이 있는지 체크하는 목적의 리스트를 만들기
visited = [False] * (n+1)
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF] * (n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
	a,b,c = map(int, input().split())
    # a번 노드에서 b번 노드로 가는 비용이 c라는 의미.
    graph[a].append((b,c))

# 방문하지 않은 노드 중에서, 가장 최단 거리가 짧은 노드의 번호를 반환
def get_smallest_node():
	min_value = INF
    index = 0  # 가장 최단 거리가 짧은 노드(인덱스)
    for i in range(1, n+1):
    	if distance[i] < min_value and not visited[i]:
        	min_value = distance[i]
            index = i
    return index
    
def dijkstra(start):
	# 시작 노드에 대해서 초기화
    distance[start] = 0
    visited[start] = True
    for j in graph[start]:
    	distance[j[0]] = j[1]
    # 시작 노드를 제외한 전체 n-1개의 노드에 대해 반복
    for i in range(n-1):
    	# 현재 최단 거리가 가장 짧은 노드를 꺼내서, 방문 처리
        now = get_smallest_node()
        visited[now] = True
        # 현재 노드와 연결된 다른 노드를 확인
        for j in graph[now]:
        	cost = distance[now]+j[1]
            # 현재 노드를 거쳐서 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[j[0]]:
            	distance[j[0]] = cost
                
# 다익스트라 알고리즘을 수행
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리를 출력
for i in range(1, n+1):
	# 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
    if distance[i] == INF:
    	print("INFINITY")
    # 도달할 수 있는 경우 거리를 출력
    else:
    	print(distance)
```

## Djikstra's algorithm: 간단한 구현 방법 Complexity

- 총 *O*(*V*)번에 걸쳐서 최단 거리가 가장 짧은 노드를 매번 선형 탐색해야 합니다.
- 따라서 전체 시간 복잡도는 *O*(*V*2)입니다.
  `이때, V는 노드의 개수를 의미한다.`
- 일반적으로 코딩 테스트의 최단 경로 문제에서는 전체 노드의 개수가 5,000개 이하라면 이 코드로 문제를 해결할 수 있습니다.
  `- 하지만 노드의 개수가 10,000개를 넘어가는 문제라면 어떻게 해야 할까요?`

# Dijkstra's Algorithm: 개선된 구현

- 단계마다 방문하지 않은 노드 중에서 최단 거리가 가장 짧은 노드를 선택하기 위해 힙(Heap) 자료구조를 이용한다.
- 다익스트라 알고리즘이 동작하는 기본 원리는 동일하다.
  `- 현재 가장 가까운 노드를 저장해 놓기 위해서 힙 자료구조를 추가적으로 이용한다는 점이 다르다.`
  `- 현재의 최단 거리가 가장 짧은 노드를 선택해야 하므로 최소 힙을 사용한다.`

## Dijkstra's Algorithm: 동작 과정 살펴보기 (우선순위 큐)

![img](https://velog.velcdn.com/images/yeahxne/post/0da3cf25-f05d-4422-82bb-a72d1982de62/image.png)

- 출발 노드를 '1번'이라고 가정한다. '1번' 노드까지의 현재 최단 거리 값을 0으로 설정해서 우선순위 큐에 넣는다.
- 우선순위 큐에 데이터를 넣을 때 튜플 형태로 데이터를 묶는 과정에서 첫번째 원소를 거리로 설정하게 되면 이 거리를 기준으로 해서 더 거리가 작은 원소가 먼저 나올 수 있도록 큐가 구성된다.

![img](https://velog.velcdn.com/images/yeahxne/post/6c95be07-f793-42cd-beac-b48f676d7f6b/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/54833468-f207-4a93-8ebb-3edaa6677cc1/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/9ba8bfe0-2167-492a-9196-b78ca69f2318/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/c1628594-3318-42ce-9de8-ab3b1a26b45f/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/e5184c8b-51d2-4ff4-8e82-336d75ec1c01/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/6e75d4fe-6820-4ef1-a857-297a55ccb5e9/image.png)

# Dijkstra's Algorithm: 개선된 구현 방법

```python
import heapq
import sys
input = sys.stdin.readline
INF = int(1e9)  # 무한을 의미하는 값으로 10억을 설정

# 노드의 개수, 간선의 개수를 입력받기
n, m = map(int, input().split())
# 시작 노드 번호 입력받기
start = int(input())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n+1)]
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF]*(n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
	a, b, c = map(int, input().split())
	# a번 노드에서 b번 노드를 가는 비용이 c라는 의미
    graph[a].append((b,c))
    
def dijkstra(start):
	q = []
    # 시작 노드로 가기 위한 최단 경로는 0으로 설정하여, 큐에 삽입
    heapq.heappush(q,(0,start))
    distance[start] = 0
    while q:  # 큐가 비어있지 않다면
    	# 가장 최단 거리가 짧은 노드에 대한 정보 꺼내기
        dist, now = heapq.heappop(q)
        # 현재 노드가 이미 처리된 적이 있는 노드라면 무시
        if distance[now] < dist:
        	continue
        # 현재 노드와 연결된 다른 인접한 노드들을 확인
        for i in graph[now]:
        	cost = dist+i[1]
            # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[i[0]]:
            	distance[i[0]] = cost
                heapq.heappush(q,(cost, i[0]))

# 다익스트라 알고리즘을 수행
dijkstra(start)

# 모든 노드로 가기 위한 최단 거리를 출력
for i in range(1, n+1):
	# 도달할 수 없는 경우, 무한(INFINITY)이라고 출력
    if distance[i] == INF:
    	print("INFINITY")
    # 도달할 수 있는 경우 거리를 출력
    else:
    	print(distance[i])
```

## Dijkstra's Algorithm: 개선된 구현 방법 Complexity

- 힙 자료구조를 이용하는 다익스트라 알고리즘의 시간 복잡도는 *O*(*E**l**o**g**V*)입니다.
- 노드를 하나씩 꺼내 검사하는 반복문(while문)은 노드의 개수 V 이상의 횟수로는 처리되지 않습니다.
  `- 결과적으로 현재 우선순위 큐에서 꺼낸 노드와 연결된 다른 노드들을 확인하는 총횟수는 최대 간선의 개수(E)만큼 연산이 수행될 수 있다.`
- 직관적으로 전체 과정은 E개의 원소를 우선순위 큐에 넣었다가 모두 빼내는 연산과 매우 유사하다.
  ![img](https://velog.velcdn.com/images/yeahxne/post/7bfbdda3-dfbb-4fa5-a680-097f5ca30508/image.png)

# Example: 전보

- 어떤 나라에는 N개의 도시가 있다. 그리고 각 도시는 보내고자 하는 메시지가 있는 경우, 다른 도시로 전보를 보내서 다른 도시로 해당 메시지를 전송할 수 있다.
- 하지만 X라는 도시에서 Y라는 도시로 전보를 보내고자 한다면, 도시 X에서 Y로 향하는 통로가 설치되어 있어야 한다. 예를 들어 X에서 Y로 향하는 통로는 있지만, Y에서 X로 향하는 통로가 없다면 Y는 X로 메시지를 보낼 수 없다. 또한 통로를 거쳐 메시지를 보낼 때는 일정 시간이 소요된다.
- 어느 날 C라는 도시에서 위급 상황이 발생했다. 그래서 최대한 많은 도시로 메시지를 보내고자 한다. 메시지는 도시 C에서 출발하여 각 도시 사이에 설치된 통로를 거쳐, 최대한 많이 퍼져나갈 것이다.
- 각 도시의 번호와 통로가 설치되어 있는 정보가 주어졌을 때, 도시 C에서 보낸 메시지를 받게 되는 도시의 개수는 총 몇 개이며 도시들이 모두 메시지를 받는 데까지 걸리는 시간은 얼마인지 계산하는 프로그램을 작성하시오.
  ![img](https://velog.velcdn.com/images/yeahxne/post/45bdb9d4-a595-42fe-a670-fab30d2a2271/image.png)

![img](https://velog.velcdn.com/images/yeahxne/post/f9da25db-6da6-41c0-9086-7d825744fd30/image.png)
`노드의 개수가 최대 30,000개, 간선의 개수가 최대 200,000개 입력될 수 있기 때문에 힙을 이용한 다익스트라 알고리즘을 작성해야 한다.`

```python
import heapq
import sys
input = sys.stdin.readline
INF = int(1e9)  # 무한을 의미하는 값으로 10억을 설정

def dijkstra(start):
	q = []
    # 시작 노드로 가기 위한 최단 거리는 0으로 설정하여, 큐에 삽입
    heapq.heappush(q, (0,start))
    distance[start] = 0
    while q:  # 큐가 비어있지 않다면
    	# 가장 최단 거리가 짧은 노드에 대한 정보를 꺼내기
        dist, now = heapq.heappop(q)
        if distance[now] < dist:
        	continue
        # 현재 노드와 연결된 다른 인접한 노드들을 확인
        for i in graph[now]:
        	cost = dist+i[1]
            # 현재 노드를 거쳐서, 다른 노드로 이동하는 거리가 더 짧은 경우
            if cost < distance[i[0]]:
            	distance[i[0]] = cost
                heapq.heappush(q, (cost, i[0]))

# 노드의 개수, 간선의 개수, 시작 노드를 입력받기
n, m, start = map(int, input().split())
# 각 노드에 연결되어 있는 노드에 대한 정보를 담는 리스트를 만들기
graph = [[] for i in range(n+1)]
# 최단 거리 테이블을 모두 무한으로 초기화
distance = [INF]*(n+1)

# 모든 간선 정보를 입력받기
for _ in range(m):
	x,y,z = map(int, input().split())
    # X번 노드에서 Y번 노드로 가는 비용이 Z라는 의미
    graph[x].append((y,z))

# 다익스트라 알고리즘을 수행
dijkstra(start)

# 도달할 수 있는 노드의 개수
count = 0
# 도달할 수 있는 노드 중에서, 가장 멀리 있는 노드와의 최단 거리
max_distance = 0
for d in distance:
	# 도달할 수 있는 노드인 경우
    if d != 1e9:
    	count += 1
        max_distance = max(max_distance, d)

# 시작 노드는 제외해야 하므로 count-1을 출력
print(count-1, max_distance)
```