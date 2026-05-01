"""Pre-order traversal: visit node, then left subtree, then right subtree.

    Order: NODE -> LEFT -> RIGHT

When to use
-----------
- Copying / serializing a tree (the root must be emitted before its
  children so a deserializer can rebuild the structure top-down).
- Polish notation / prefix expression output.

Algorithm sketches
------------------
Recursive (clearest):
    def preorder(node, out):
        if node is None: return
        out.append(node.value)
        preorder(node.left, out)
        preorder(node.right, out)

Iterative with an explicit stack (avoid Python's recursion limit on
deep trees):
    stack = [root]
    while stack:
        node = stack.pop()
        if node is None: continue
        out.append(node.value)
        # push RIGHT first so LEFT is processed first
        stack.append(node.right)
        stack.append(node.left)

Things to research
------------------
- Morris traversal - O(1) extra space by temporarily rewriting links.
- Why iterative DFS pushes children in reverse order.
"""


def preorder():
    raise NotImplementedError()
