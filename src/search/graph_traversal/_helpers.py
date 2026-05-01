"""Shared utilities for graph traversal algorithms."""

from __future__ import annotations

from collections.abc import Hashable, Mapping


def reconstruct_path(
    parents: Mapping[Hashable, Hashable | None],
    start: Hashable,
    goal: Hashable,
) -> list[Hashable]:
    """Reconstruct the path from ``start`` to ``goal`` from a predecessor map.

    BFS, DFS, Dijkstra, and A* all build the same artifact during their
    traversal: a ``{node: predecessor}`` map where ``parents[start] is None``.
    This helper walks that map backwards from ``goal`` to ``start`` and
    returns the path in forward order.

    Returns an empty list when ``goal`` is unreachable (i.e. ``goal`` is not
    a key in ``parents`` and is not equal to ``start``). Raises ``ValueError``
    if the predecessor chain is malformed (cycle or detached from ``start``).
    """
    if goal == start:
        return [start]
    if goal not in parents:
        return []

    path: list[Hashable] = []
    seen: set[Hashable] = set()
    node: Hashable | None = goal
    while node is not None:
        if node in seen:
            raise ValueError(f"cycle in predecessor map at node {node!r}")
        seen.add(node)
        path.append(node)
        if node == start:
            path.reverse()
            return path
        node = parents.get(node)

    raise ValueError(f"predecessor chain from {goal!r} does not reach {start!r}")
