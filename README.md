# DOLCE

Materials for ""DOLCE: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects"**.

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

## Citation
```
@misc{tamano2025dolce,
    title = {{DOLCE}: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects},
    publisher = {arXiv},
    author = {Tamano, Shu and Nojima, Masanori},
    year = {2025},
    note = {arXiv:2505.00961},
}
```

## Contact

If you have any question, please feel free to contact: stamano@niid.go.jp