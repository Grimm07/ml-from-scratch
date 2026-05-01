# ml-from-scratch

A learning repository for implementing machine-learning, search, and
probability building blocks from scratch. Everything is scaffolded as
placeholders with guiding comments - the *implementations* are exercises
for you to fill in.

## Overview

Three things live here:

1. **`src/`** - source-code placeholders for algorithms and models
   (linear regression, MLP, CNN, transformer, GAN, K-Means, PCA, Bayes
   nets, BFS/DFS/Dijkstra/A*, minimax, simulated annealing, tree
   traversals, ...). Each module raises `NotImplementedError` and
   carries a docstring with: an algorithm sketch, the API it should
   expose, hints about pitfalls, and pointers to topics worth
   researching before writing the code.

2. **`tests/`** - pytest test scaffolds for every model in `src/`.
   Tests use `pytest.importorskip` so the suite stays green while
   modules are placeholders, and `xfail(strict=False)` so existing
   `NotImplementedError` stubs don't fail CI. Datasets are loaded via
   shared fixtures in `tests/conftest.py` (sklearn, torchvision,
   HuggingFace, plus an in-memory Tiny Shakespeare corpus).

3. **`problems/`** - worked-problem placeholders for probability,
   statistics, and optimization. Each file states a problem, defines
   the relevant concepts, and hints at the solution strategy without
   giving the answer. You implement two functions per problem - usually
   one analytic and one Monte-Carlo - and verify they agree.

## Architecture

```
src/
  supervised/
    linear_models/linear_regression.py
  search/
    graph_traversal/
      bfs.py, dfs.py, dijkstra.py, a_star.py
      bidirectional.py, tridirectional.py
      _helpers.py            # reconstruct_path (only real implementation)
    adversarial/
      minimax.py, alpha_beta.py
    local_search/
      simulated_annealing.py
  tree/
    traversal/
      preorder.py, inorder.py, postorder.py, level_order.py
  probabilistic/
    bayes_nets.py

tests/
  conftest.py                # shared dataset fixtures
  test_supervised/           # linear/logistic, GBM, MLP, CNN, transformer
  test_unsupervised/         # K-Means, PCA
  test_generative/           # GAN

problems/
  probability/               # conditional & joint, Bayes' theorem, Monty Hall
  stats/                     # MLE/MAP, bias-variance, central limit theorem
  optimization/              # simulated annealing on TSP
```

## Setup

This project uses **`uv`** as the dependency manager and environment
runner, and **`hatchling` + `hatch-vcs`** as the build backend.

```bash
# create the .venv and install runtime + dev deps
uv sync

# activate the venv (optional - uv run does this for you)
source .venv/bin/activate
```

Python >= 3.11 is required.

## Testing

```bash
# run the whole suite
uv run pytest

# skip the slow tests that train on real datasets
uv run pytest -m "not slow"

# run one model's tests
uv run pytest tests/test_supervised/test_linear_regression.py -v
```

Image-dataset tests (MNIST, FashionMNIST, CIFAR-10) are skipped unless
the data is cached locally. Set `MLFS_ALLOW_DOWNLOAD=1` to let them
download on first run.

## Working through the problems

Each file under `problems/` is runnable as a script. Before you've
implemented the stubs, running it will raise `NotImplementedError` at
the first unimplemented function - that's your TODO list.

```bash
uv run python problems/probability/monty_hall.py
uv run python problems/stats/mle_and_map.py
uv run python problems/optimization/simulated_annealing.py
```

A typical workflow:

1. Read the module docstring - it lays out the problem, the relevant
   probability/statistics concepts, and what to research.
2. Read each function's docstring for hints on the strategy without
   spoiling the answer.
3. Implement the analytic solution first, then verify it against the
   Monte-Carlo simulation. If they disagree, one of them has a bug -
   diagnosing which is part of the exercise.

Recommended order:

| Order | File | Concept |
|-------|------|---------|
| 1 | `problems/probability/conditional_and_joint.py` | joint / marginal / conditional, chain rule |
| 2 | `problems/probability/bayes_theorem.py` | Bayes, sequential updating, Naive Bayes |
| 3 | `problems/probability/monty_hall.py` | conditioning on a sampling rule |
| 4 | `problems/stats/mle_and_map.py` | MLE, MAP, conjugate priors, regularization |
| 5 | `problems/stats/clt.py` | sample mean -> Gaussian, 1/sqrt(n) shrinkage |
| 6 | `problems/stats/bias_variance.py` | underfit / overfit / irreducible noise |
| 7 | `problems/optimization/simulated_annealing.py` | metaheuristics, MCMC connection |

## Implementing the source modules

Once the problems make sense, the `src/` modules are next. Each one
has a docstring with:

- An **algorithm sketch** (pseudocode) so you know the shape.
- A list of **things to research** so you can dig in.
- Pointers to the relevant test file in `tests/`, which doubles as the
  spec for the module's API.

Suggested order, easiest first:

1. `src/search/graph_traversal/{bfs,dfs}.py` - shortest practical algorithms.
2. `src/tree/traversal/*.py` - smaller scope; gentle warmup.
3. `src/search/graph_traversal/{dijkstra,a_star}.py` - priority-queue mechanics.
4. `src/supervised/linear_models/linear_regression.py` - first ML model.
5. `src/search/local_search/simulated_annealing.py` - ties to the SA exercise.
6. `src/probabilistic/bayes_nets.py` - capstone for the probability track.
7. Adversarial search, bidirectional/tridirectional, the deep-learning
   models (MLP, CNN, transformer, GAN) - each is a project on its own.

## Resources

- **Probability / statistics**: Bishop, *Pattern Recognition and Machine
  Learning*, ch. 1-2. Wasserman, *All of Statistics*.
- **Search / AI**: Russell & Norvig, *AI: A Modern Approach*, ch. 3-5.
- **Deep learning**: Goodfellow, Bengio, Courville, *Deep Learning*.
  Karpathy's "Neural Networks: Zero to Hero" video series for
  hands-on transformer / GAN work.
