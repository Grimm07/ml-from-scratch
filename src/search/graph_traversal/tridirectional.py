"""Tridirectional search.

Idea
----
Generalization of bidirectional search to *three* sources. You're
typically asked: "find the minimum-cost set of paths that connects
nodes A, B, and C", which is a small Steiner-tree problem.

Run three frontiers - one rooted at each of A, B, C - and track the
*pairwise* meeting events. The optimal connector is the cheapest of:

    a) Two pairs meet, one node is reached via that pair: e.g.
       cost(A-B path) + cost(B-C path), with B being a transit.
    b) All three frontiers meet at a single intermediate node M:
       cost(A-M) + cost(B-M) + cost(C-M).

You return the configuration with minimum total cost.

Where this comes up
-------------------
- The Berkeley AI / Georgia Tech "AI" courses use this on a Romania-map
  type problem - it's a classic graduate-level homework.
- Multi-agent path planning ("everyone meet up cheaply").

Things to research
------------------
- Why naive "stop on first meeting" can be wrong for weighted graphs.
- The correct termination condition: keep expanding until the *minimum*
  current frontier key exceeds the best stitched solution found so far.
- The Steiner-tree connection - tridirectional search is a special case.
"""


def tridirectional_search():
    raise NotImplementedError()
