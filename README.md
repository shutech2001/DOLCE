# DOLCE

Materials for "[**DOLCE: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects**](https://arxiv.org/abs/2505.00961)".

## What is this repo?

This repository includes an implementation of DOLCE, a novel off-policy evaluation and learning method for violating common and full support assumption.
It also contains the numerical experiments presented in the paper.

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

### Main file
- `src/ope/models.py`
  - function of `calc_dolce` is ours
- `src/opl/models.py`
  - class of `DOLCE` is ours

## Citation
```
@article{tamano2025dolce,
    author={Tamano, Shu},
    title={{DOLCE}: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects},
    journal={arXiv preprint arXiv:2505.00961},
    year={2025},
}
```

## Contact

If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp