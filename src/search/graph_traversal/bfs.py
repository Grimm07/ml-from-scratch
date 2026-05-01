"""Breadth-first search.

Algorithm sketch
----------------
    frontier = FIFO queue, initialized with [start]
    visited  = {start}
    parents  = {start: None}
    while frontier not empty:
        node = frontier.popleft()
        if node == goal: stop
        for nbr in neighbors(node):
            if nbr not in visited:
                visited.add(nbr)
                parents[nbr] = node
                frontier.append(nbr)

Properties
----------
- Optimal for *unweighted* graphs (returns shortest hop-count path).
- O(V + E) time and space.
- The {node: parent} dict can be passed to ``_helpers.reconstruct_path``
  to recover the path.

Things to research
------------------
- Why a FIFO (deque) and not a stack (which would give DFS).
- Iterative-deepening DFS as an alternative when memory is tight.
- "Bidirectional BFS" (see bidirectional.py) - same idea, two frontiers.
"""


def bfs():
    raise NotImplementedError()
