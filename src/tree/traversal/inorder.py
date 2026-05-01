"""In-order traversal: left subtree, then node, then right subtree.

    Order: LEFT -> NODE -> RIGHT

Key fact
--------
On a *binary search tree* (BST), in-order traversal yields keys in
sorted ascending order. This is the fastest way to verify a candidate
BST or to extract a sorted sequence from one.

Algorithm sketches
------------------
Recursive:
    def inorder(node, out):
        if node is None: return
        inorder(node.left, out)
        out.append(node.value)
        inorder(node.right, out)

Iterative with a stack (a bit subtler than pre-order):
    stack = []
    node = root
    while stack or node is not None:
        while node is not None:
            stack.append(node)
            node = node.left
        node = stack.pop()
        out.append(node.value)
        node = node.right

Things to research
------------------
- "Validate BST" interview question - in-order makes it a one-liner
  (check the sequence is strictly increasing).
- Threaded binary trees - a data-structure trick that makes in-order
  traversal O(1) per step without recursion or a stack.
"""


def inorder():
    raise NotImplementedError()
