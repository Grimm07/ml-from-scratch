"""Minimax search for two-player zero-sum games.

Setting
-------
Two players (call them MAX and MIN) alternate moves. MAX wants to
maximize a numeric utility evaluated at terminal states; MIN wants to
minimize it. Minimax computes the *backed-up* value of each non-terminal
state assuming both players play optimally.

Algorithm sketch (recursive)
----------------------------
    def minimax(state, depth, maximizing):
        if depth == 0 or terminal(state):
            return evaluate(state)
        if maximizing:
            best = -inf
            for s in successors(state):
                best = max(best, minimax(s, depth-1, False))
            return best
        else:
            best = +inf
            for s in successors(state):
                best = min(best, minimax(s, depth-1, True))
            return best

Properties
----------
- Optimal under the (often unrealistic) assumption that both players
  play optimally.
- Time complexity: O(b^d) where b = branching factor, d = depth.
- Space complexity: O(d) due to the recursion stack.

Things to research / decide
---------------------------
- "Negamax" - a cleaner formulation where you negate the score and only
  ever maximize. Reduces if/else duplication.
- Where the *evaluation function* comes from for games where you can't
  search to terminal states (chess, Go) - hand-crafted vs learned.
- How to break ties / order moves to make alpha-beta pruning effective
  (see alpha_beta.py - move ordering matters enormously).

Connection to ML
----------------
Reinforcement learning's "value function" is the same idea generalized
to single-player MDPs and learned rather than hand-coded. AlphaGo et al
combine minimax-style tree search with a neural-network evaluator.
"""


def minimax():
    raise NotImplementedError()
