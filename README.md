# DOLCE

Materials for "[**DOLCE: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects**](https://arxiv.org/abs/2505.00961)".

## What is this repo?

This repository provides:
- An implementation of DOLCE for off-policy evaluation (OPE) and off-policy learning (OPL) under support violation.
- Synthetic experiments used in the paper.
- A real-world data processing / evaluation pipeline (code only; the data itself is confidential).

## Requirements and Setup
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

## Repository Layout (Key Files)
- `src/ope/models.py`: OPE estimators (`calc_dm`, `calc_ips`, `calc_dr`, `calc_dolce`)
- `src/opl/models.py`: OPL learners (`RegressionBasedPolicyLearner`, `GradientBasedPolicyLearner`, `DOLCE`)
- `src/synthetic/`: synthetic data generator and ground-truth utilities
- `scripts/`: experiment drivers
  - `scripts/exec_synthetic_ope.py`
  - `scripts/exec_synthetic_opl.py`
  - `scripts/exec_real_world.py` (pipeline entrypoint)
  - `scripts/real_world/` (pipeline components)

## Run DOLCE Only (Function/Class Level)

### OPE: `calc_dolce`
You can run the DOLCE estimator alone (without the full experiment scripts) as follows:
```python
from synthetic import generate_synthetic_data
from ope import calc_dolce
from utils import eps_greedy_policy

data = generate_synthetic_data(
    num_data=1000,
    num_features=5,
    num_actions=5,
    non_overlap_ratio=0.3,
    lambda_=0.5,
    eta=0.0,
    random_state=42,
    env_random_state=7,
)

# Target policy (example: epsilon-greedy on current-context component)
q_for_pi = data.get("g_x_t_a_t", data["q"])
pi = eps_greedy_policy(q_for_pi)

contrib, info = calc_dolce(data, pi, random_state=42)
print("DOLCE estimate:", float(contrib.mean()))
print("lag weight range:", float(info["lag_weight_min"]), float(info["lag_weight_max"]))
```

### OPL: `DOLCE` learner
You can also run the DOLCE policy learner directly:
```python
from synthetic import generate_synthetic_data
from opl import DOLCE

logged = generate_synthetic_data(
    num_data=1000,
    num_features=5,
    num_actions=5,
    non_overlap_ratio=0.3,
    lambda_=0.5,
    eta=0.0,
    random_state=42,
    env_random_state=7,
    logging_eps=0.2,
)
test = generate_synthetic_data(
    num_data=2000,
    num_features=5,
    num_actions=5,
    non_overlap_ratio=0.3,
    lambda_=0.5,
    eta=0.0,
    random_state=999,
    env_random_state=7,
    logging_eps=0.2,
)

learner = DOLCE(num_features=5, num_actions=5, max_iter=30, random_state=42)
learner.fit(logged, test)
pi_hat = learner.predict(test)
value = float((test["q"] * pi_hat).sum(1).mean())
print("Learned policy value:", value)
```

## Experiments (Synthetic Data Only)

The synthetic experiments are controlled by the two scripts below. Each script supports sweeping:
- `support violation ratio` (default)
- `logged data size`
- `number of actions`
- `lambda`
- `eta`

### OPE experiments
```shell
python3 scripts/exec_synthetic_ope.py --sweep support_violation
python3 scripts/exec_synthetic_ope.py --sweep num_data
python3 scripts/exec_synthetic_ope.py --sweep num_actions
python3 scripts/exec_synthetic_ope.py --sweep lambda
python3 scripts/exec_synthetic_ope.py --sweep eta
```

### OPL experiments
```shell
python3 scripts/exec_synthetic_opl.py --sweep support_violation
python3 scripts/exec_synthetic_opl.py --sweep num_data
python3 scripts/exec_synthetic_opl.py --sweep num_actions
python3 scripts/exec_synthetic_opl.py --sweep lambda
python3 scripts/exec_synthetic_opl.py --sweep eta
```

By default, each `--sweep` uses the paper's grid; you can override with `--values "..."` if needed.
When sweeping anything other than support violation, you can fix the ratio with `--support-violation` (default: 0.0).
Plots are saved under `results/plots/` as `ope_<sweep>.pdf` / `opl_<sweep>.pdf`.

## Real-World Pipeline (Code Only)

The real-world dataset is confidential, so this repository provides **only the processing/evaluation pipeline**.
Relevant files:
- `scripts/exec_real_world.py` (entrypoint)
- `scripts/real_world/` (data loading, preprocessing, and OPE evaluation)

## Citation
```text
@article{tamano2025dolce,
    author={Tamano, Shu},
    title={{DOLCE}: Decomposing Off-Policy Evaluation/Learning into Lagged and Current Effects},
    journal={arXiv preprint arXiv:2505.00961},
    year={2025},
}
```

## Contact

If you have any question, please feel free to contact: tamano-shu212@g.ecc.u-tokyo.ac.jp
