"""Bayesian network: directed acyclic graph of conditional probability tables.

A Bayesian network factorizes a joint distribution as

    P(X_1, ..., X_n) = prod_i P(X_i | parents(X_i))

The graph encodes conditional independence assumptions; the CPTs
(conditional probability tables) supply the actual numbers.

Eventual API sketch
-------------------
    net = BayesNet()
    net.add_node("Rain", parents=[], cpt={(): {True: 0.2, False: 0.8}})
    net.add_node(
        "Sprinkler",
        parents=["Rain"],
        cpt={(True,): {True: 0.01, False: 0.99},
             (False,): {True: 0.4,  False: 0.6}},
    )
    net.add_node(
        "WetGrass",
        parents=["Rain", "Sprinkler"],
        cpt={(True, True):   {True: 0.99, False: 0.01},
             (True, False):  {True: 0.8,  False: 0.2},
             (False, True):  {True: 0.9,  False: 0.1},
             (False, False): {True: 0.0,  False: 1.0}},
    )

    net.joint({"Rain": True, "Sprinkler": False, "WetGrass": True})
    net.query("Rain", evidence={"WetGrass": True})
    net.sample(n=1000, evidence={"WetGrass": True})

Things to research while building each method
---------------------------------------------
- ``add_node``: validate that all named parents already exist and that
  the CPT has one row per Cartesian product of parent values, with each
  row summing to 1.0 within float tolerance.
- ``joint``: walk nodes in topological order; multiply
  P(node | parents) values pulled from each CPT. Easy O(n) per call.
- ``query`` (exact inference): start with "enumeration" - it's O(2^n)
  but the simplest to get right; worth implementing first to have a
  ground truth to test variable elimination against later.
- ``sample`` (approximate inference): three classic approaches, each
  worth its own subfunction:
    1. Rejection sampling: generate samples from the joint, throw away
       ones that don't match the evidence. Wasteful for unlikely evidence.
    2. Likelihood weighting: fix evidence vars to their observed values,
       sample non-evidence vars, weight each sample by P(evidence | parents).
    3. Gibbs sampling (MCMC): cycle through non-evidence vars,
       resampling each from its conditional given the current values of
       its Markov blanket.

Tests to keep in mind
---------------------
- Verify a small hand-computed joint matches.
- Verify ``query`` against an analytic posterior on a 3-node sprinkler
  network (the standard textbook example).
- Verify approximate samplers converge to the exact posterior as
  n -> infinity (within sampling noise).

Connection to the problems/ directory
-------------------------------------
- problems/probability/conditional_and_joint.py builds the *intuition*
  for joint and conditional probabilities by hand. A Bayes net is the
  scalable, structured way to assemble those into a full model.
- problems/probability/bayes_theorem.py is a 2-node Bayes net (D, T)
  done by hand. Generalizing it to arbitrary DAGs is what lives here.
"""


class BayesNet:
    def __init__(self):
        # Suggested internal state:
        #   self.nodes: list of node names in insertion (topological) order
        #   self.parents: dict[name, list[name]]
        #   self.cpts:    dict[name, dict[tuple_of_parent_values, dict[value, prob]]]
        raise NotImplementedError()

    def add_node(self, name, parents, cpt):
        """Add a node with given parents and CPT.

        Validation hints:
        - parents must already exist (insertion order is the topo sort).
        - cpt keys must enumerate every assignment of parent values.
        - each cpt row (a dict of value -> prob) must sum to 1 +/- eps.
        """
        raise NotImplementedError()

    def joint(self, assignment):
        """P(X1=x1, ..., Xn=xn) - product of CPT entries.

        Hint: for each node, look up its parents' values in `assignment`,
        then multiply by cpts[node][parent_values_tuple][assignment[node]].
        """
        raise NotImplementedError()

    def query(self, variable, evidence):
        """P(variable | evidence). Start with enumeration (brute force):

            for each value v of `variable`:
                numerator[v] = sum over assignments of hidden vars of
                               joint(value=v, evidence, hidden)
            normalize numerator and return as a dict.
        """
        raise NotImplementedError()

    def sample(self, n, evidence=None):
        """Draw n samples from the posterior given evidence.

        Implement progressively:
            1. Rejection sampling   (simplest; correct but slow for rare evidence)
            2. Likelihood weighting (no rejection; weighted samples)
            3. Gibbs sampling       (MCMC; needs Markov-blanket conditional)
        """
        raise NotImplementedError()
