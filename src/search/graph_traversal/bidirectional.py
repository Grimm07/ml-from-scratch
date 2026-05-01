"""Bidirectional search.

Idea
----
Run two BFS frontiers simultaneously - one expanding outward from
``start``, the other expanding *backward* from ``goal`` - and stop the
moment they meet at any common node.

Why it's faster
---------------
Single BFS visits ~b^d nodes where b = branching factor and d = depth.
Two BFS frontiers each visit ~b^(d/2). Sum is 2 * b^(d/2), which is
exponentially smaller than b^d for any non-trivial b and d.

Subtleties to research
----------------------
- The "meeting" check has to happen *before* expanding a node, not
  after, otherwise you can miss the optimal meeting point.
- For *weighted* graphs the analogous algorithm is bidirectional
  Dijkstra (sometimes called "front-to-back" or "front-to-front"
  variants); the termination condition is more delicate (the optimum
  may not pass through the *first* meeting node).
- Reverse adjacency: bidirectional needs to walk edges *in reverse*
  from the goal side. For undirected graphs this is free; for directed
  graphs you need a transposed adjacency representation.

Algorithm sketch
----------------
    forward_frontier  = deque([start]);  forward_parents  = {start: None}
    backward_frontier = deque([goal]);   backward_parents = {goal:  None}
    while both frontiers non-empty:
        expand one level on the smaller frontier
        if any node visited by both frontiers:
            stitch the two parent maps and return the path
"""


def bidirectional_search():
    raise NotImplementedError()
