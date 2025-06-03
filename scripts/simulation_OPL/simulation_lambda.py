import pickle
from tqdm import tqdm
import warnings

import torch
import pandas as pd
from sklearn.utils import check_random_state

from OPL.generate_dataset import generate_synthetic_data
from OPL.estimate_reward import train_reward_model
from OPL.policy_learners import (
    RegressionBasedPolicyLearner,
    GradientBasedPolicyLearner,
    DOLCE,
)

warnings.filterwarnings("ignore")


def main():
    # Parameters
    num_data: int = 1000
    num_actions: int = 5
    non_overlap_ratio: float = 0.1
    num_simulation: int = 100
    num_features: int = 5
    num_epochs: int = 30
    test_data_size: int = 50000
    random_state: int = 42
    torch.manual_seed(random_state)
    _ = check_random_state(random_state)

    lambda_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    result_df_list = []
    for lambda_ in lambda_list:
        test_data = generate_synthetic_data(
            num_data=test_data_size,
            num_features=num_features,
            num_actions=num_actions,
            non_overlap_ratio=non_overlap_ratio,
            lambda_=lambda_,
        )
        pi_0_value = (test_data["q"] * test_data["pi_0"]).sum(1).mean()
        test_policy_value_list = []
        for sim in tqdm(range(num_simulation), desc=f"lambda={lambda_}..."):
            logged_data = generate_synthetic_data(
                num_data=num_data,
                num_features=num_features,
                num_actions=num_actions,
                non_overlap_ratio=non_overlap_ratio,
                lambda_=lambda_,
                random_state=sim * 100,
            )

            true_value_of_learned_policies = dict()
            true_value_of_learned_policies["logging"] = pi_0_value

            # DM
            reg = RegressionBasedPolicyLearner(
                num_features=num_features,
                num_actions=num_actions,
                max_iter=num_epochs,
                random_state=sim * 100,
            )
            reg.fit(logged_data, test_data)
            pi_reg = reg.predict(test_data)
            true_value_of_learned_policies["reg"] = (test_data["q"] * pi_reg).sum(1).mean()
            # IPS
            ips = GradientBasedPolicyLearner(
                num_features=num_features,
                num_actions=num_actions,
                max_iter=num_epochs,
                random_state=sim * 100,
            )
            ips.fit(logged_data, test_data)
            pi_ips = ips.predict(test_data)
            true_value_of_learned_policies["ips"] = (test_data["q"] * pi_ips).sum(1).mean()
            # DR
            dr = GradientBasedPolicyLearner(
                num_features=num_features,
                num_actions=num_actions,
                max_iter=num_epochs,
                random_state=sim * 100,
            )
            reg = RegressionBasedPolicyLearner(
                num_features=num_features,
                num_actions=num_actions,
                max_iter=num_epochs,
                random_state=sim * 100,
            )
            reg.fit(logged_data, test_data)
            q_hat = reg.predict_q(logged_data)
            dr.fit(logged_data, test_data, q_hat=q_hat)
            pi_dr = dr.predict(test_data)
            true_value_of_learned_policies["dr"] = (test_data["q"] * pi_dr).sum(1).mean()
            # DOLCE
            dolce = DOLCE(
                num_features=num_features,
                num_actions=num_actions,
                max_iter=num_epochs,
                random_state=sim * 100,
            )
            q_hat = train_reward_model(dataset=logged_data)
            dolce.fit(logged_data, test_data, q_hat=q_hat)
            pi_dolce = dolce.predict(test_data)
            true_value_of_learned_policies["dolce"] = (
                (test_data["q"] * pi_dolce).sum(1).mean()
            )

            test_policy_value_list.append(true_value_of_learned_policies)

        result_df = (
            pd.DataFrame(test_policy_value_list)
            .stack()
            .reset_index(1)
            .rename(columns={"level_1": "method", 0: "value"})
        )
        result_df["num_data"] = num_data
        result_df["num_action"] = num_actions
        result_df["non_overlap_ratio"] = non_overlap_ratio * 100
        result_df["pi_0_value"] = pi_0_value
        result_df["lambda"] = lambda_
        result_df["rel_value"] = result_df["value"] / pi_0_value
        result_df_list.append(result_df)

    result_df_lambda = pd.concat(result_df_list).reset_index(level=0)
    with open("./result_df_lambda.pkl", "wb") as f:
        pickle.dump(result_df_lambda, f)


if __name__ == "__main__":
    main()
