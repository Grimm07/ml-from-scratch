"""Depth-first search.

Algorithm sketch
----------------
    Recursive form (clearest, but careful with deep graphs / Python's
    default recursion limit ~1000):

        visited = set()
        def visit(node):
            visited.add(node)
            for nbr in neighbors(node):
                if nbr not in visited:
                    parents[nbr] = node
                    visit(nbr)

    Iterative form using a *stack* (preferred for large graphs):
        stack = [start]
        ...

Properties
----------
- O(V + E) time and space.
- *Not* optimal for shortest paths - it returns *some* path, not the
  shortest one.
- Useful for: cycle detection, topological sort, connected components,
  finding strongly connected components (Tarjan / Kosaraju).

Things to research
------------------
- "Pre-order" vs "post-order" DFS - controls when you "process" a node
  relative to its descendants. Post-order DFS gives you topological
  sort for free.
- Why iterative DFS with a stack is *not* a perfect mirror of BFS with
  a queue (children visit order differs from naive recursive DFS).
"""


def dfs():
    raise NotImplementedError()
