"""A* search with an admissible heuristic.

Algorithm sketch
----------------
    A* is Dijkstra with a smarter priority key. Instead of ordering the
    frontier by ``g(n)`` (cost so far), it uses

        f(n) = g(n) + h(n)

    where ``h(n)`` is a *heuristic estimate* of the remaining cost from
    n to the goal.

Required properties of the heuristic
------------------------------------
- *Admissible*: h(n) <= true cost from n to goal. Guarantees optimality.
- *Consistent (monotone)*: h(n) <= cost(n, n') + h(n') for every edge.
  Implies admissibility AND that nodes don't need to be re-expanded.

Examples
--------
- Grid pathfinding with 4-connected moves: h = Manhattan distance.
- Grid pathfinding with 8-connected moves: h = Chebyshev distance
  (or octile distance for unit/sqrt(2) costs).
- Euclidean h is admissible for any planar problem; not always tight.

Things to research
------------------
- Why h = 0 reduces A* to Dijkstra exactly.
- "Inadmissible heuristic" tradeoffs: faster, no longer optimal.
- IDA* (iterative deepening A*) when the search tree is huge.
"""


def a_star():
    raise NotImplementedError()
