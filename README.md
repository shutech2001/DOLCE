# DOLCE

Materials for "[**DOLCE: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects**](https://arxiv.org/abs/2505.00961)".

## What is this repo?

This repository includes an implementation of DOLCE, a novel off-policy evaluation and learning method for violating common and full support assumption.
It also contains the numerical experiments presented in the paper.

**(Note) This repository and its code are a work in progress. We will update them with improved code as it becomes available.**

### Requirements and Setup
```
# clone the repository
git clone git@github.com:shutech2001/DOLCE.git

# build the environment with poetry
poetry install

# activate virtual environment
eval $(poetry env activate)

# [Option] to activate the interpreter, select the following output as the interpreter.
poetry env info --path
```

### Executing simulations
- `python scripts/simulation_OPE/simulation_{}.py`
  - if `{} == action_num`, then varying the number of actions
  - if `{} == lambda`, then varying balancing parameter of lagged and current effects
  - if `{} == non_overlap`, then varying the ratio of out-of-support
  - if `{} == train_data`, then varying the number of data
- `python scripts/simulation_OPL/simulation_{}.py`
  - if `{} == action_num`, then varying the number of actions
  - if `{} == lambda`, then varying balancing parameter of lagged and current effects
  - if `{} == non_overlap`, then varying the ratio of out-of-support
  - if `{} == train_data`, then varying the number of data

### Main file
- `scripts/OPE/estimators.py`
  - function of `calc_dolce` is ours
- `scripts/OPL/policy_learners.py`
  - class of `DOLCE` is ours

## Citation
```
@article{tamano2025dolce,
    author={Tamano, Shu and Nojima, Masanori},
    title={{DOLCE}: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects},
    journal={arXiv preprint arXiv:2505.00961},
    year={2025},
}
```

## Contact

If you have any question, please feel free to contact: tamano.s@jihs.go.jp