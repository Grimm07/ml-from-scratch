"""Post-order traversal: left subtree, right subtree, then node.

    Order: LEFT -> RIGHT -> NODE

When to use
-----------
- Operations that must visit children before the parent: computing
  subtree sizes, summing values, freeing nodes in C/C++.
- Evaluating expression trees (children are operands, parent is the
  operator that combines them - you need the operands first).
- Topological sort on a DAG (post-order DFS, then reverse).

Algorithm sketches
------------------
Recursive (clearest):
    def postorder(node, out):
        if node is None: return
        postorder(node.left, out)
        postorder(node.right, out)
        out.append(node.value)

Iterative (trickier than pre-order):
    Two common tricks:
    1. Modified pre-order with two stacks: do "node, right, left",
       collect into stack 2, then pop stack 2 - it'll be in post-order.
    2. Single-stack with a "last visited" pointer to know when to pop.

Things to research
------------------
- Why iterative post-order is annoying compared to pre/in-order.
- Tail-call elimination tricks; not relevant in CPython but worth
  understanding.
- "Euler tour" of a tree - one structure that lets you derive any of
  the three orders by relabeling.
"""


def postorder():
    raise NotImplementedError()
