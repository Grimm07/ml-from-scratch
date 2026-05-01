"""Alpha-beta pruning - minimax with provably-irrelevant branches skipped.

The idea
--------
While minimax is running, you carry two bounds:
    alpha = best score MAX can already guarantee somewhere on the path.
    beta  = best score MIN can already guarantee somewhere on the path.

If at any node alpha >= beta, the current player can prove the *other*
player will avoid this subtree entirely - so you stop exploring it.
This is the "cutoff".

Algorithm sketch (negamax form is cleanest)
-------------------------------------------
    def alpha_beta(state, depth, alpha, beta, color):
        if depth == 0 or terminal(state):
            return color * evaluate(state)
        value = -inf
        for s in ordered_successors(state):
            value = max(value, -alpha_beta(s, depth-1, -beta, -alpha, -color))
            alpha = max(alpha, value)
            if alpha >= beta:
                break             # cutoff
        return value

Properties
----------
- Returns *the same answer* as minimax (alpha-beta is exact).
- Best-case time complexity: O(b^(d/2)) - exponentially faster than
  minimax. Worst-case: same as minimax (O(b^d)) when move ordering is
  adversarial.
- Move ordering is critical: try the most promising moves first.

Things to research
------------------
- Why -beta and -alpha swap roles in the recursive call (the clever
  bit of negamax).
- "Killer moves" / "history heuristic" / iterative deepening - common
  techniques used to get good move ordering cheaply.
- Why you can prune *more* with a tighter (smaller) initial window
  ("aspiration windows", "principal-variation search").
"""


def alpha_beta():
    raise NotImplementedError()
