"""Level-order (breadth-first) traversal of a tree.

    Visit all nodes at depth 0, then all at depth 1, then depth 2, ...

Algorithm sketch
----------------
Use a FIFO queue (collections.deque):

    queue = deque([root])
    while queue:
        node = queue.popleft()
        out.append(node.value)
        if node.left  is not None: queue.append(node.left)
        if node.right is not None: queue.append(node.right)

Variants worth implementing
---------------------------
- Group output by level: track a "level size" each iteration and pop
  exactly that many nodes before starting the next level. Useful for
  problems like "print tree level by level" or "right-side view".
- Zigzag / spiral order: reverse alternate levels.
- Bottom-up level order: same algorithm, then reverse the list of levels.

Things to research
------------------
- Why this is exactly graph BFS specialized to trees - same skeleton,
  same complexity (O(n) time, O(width) space where width = max nodes
  on any level).
- Connection to "breadth-first tree" / "level set" terminology in
  search and game-tree literature.
"""


def level_order():
    raise NotImplementedError()
