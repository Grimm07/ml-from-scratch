"""Dijkstra's shortest-path algorithm.

Algorithm sketch
----------------
    dist = {start: 0}, all others = +inf
    parents = {start: None}
    pq = min-priority-queue keyed by tentative distance, seeded with (0, start)
    while pq not empty:
        d, node = pq.popmin()
        if d > dist[node]: continue          # stale entry, skip
        if node == goal: stop
        for nbr, w in neighbors(node):
            new_d = d + w
            if new_d < dist.get(nbr, inf):
                dist[nbr] = new_d
                parents[nbr] = node
                pq.push((new_d, nbr))

Properties
----------
- Optimal for graphs with *non-negative* edge weights.
- O((V + E) log V) with a binary heap; O(E + V log V) with Fibonacci.
- Fails on negative edges - use Bellman-Ford instead.

Things to research / decide
---------------------------
- ``heapq`` is a min-heap; tuples (distance, node) sort lexicographically.
- The "lazy deletion" pattern: instead of decrease-key, push duplicates
  and skip stale ones (the `d > dist[node]` check above).
- A* (see a_star.py) is Dijkstra + heuristic - same skeleton, different
  priority key.
"""


def dijkstra():
    raise NotImplementedError()
