import pickle
from tqdm import tqdm
import warnings

import pandas as pd

from OPE.generate_dataset import generate_synthetic_data, calc_true_value
from OPE.estimate_reward import fit_predict_by_MLP_for_all_actions
from OPE.estimators import calc_dm, calc_ips, calc_dr, calc_dolce
from OPE.utils import aggregate_simulation_results, eps_greedy_policy

warnings.filterwarnings("ignore")


def main():
    # Parameters
    num_simulation: int = 100
    num_features: int = 5

    non_overlap_ratio: float = 0.5
    num_data: int = 1000
    lambda_: float = 0.5

    result_df_list = []
    num_actions_list = [2, 5, 10, 50, 100]
    for num_actions in num_actions_list:
        true_value = calc_true_value(
            num_features=num_features,
            num_actions=num_actions,
            non_overlap_ratio=non_overlap_ratio,
            lambda_=lambda_,
        )

        estimated_policy_value_list = []
        for sim in tqdm(range(num_simulation), desc=f"num_actions={num_actions}"):
            logged_data = generate_synthetic_data(
                num_data=num_data,
                num_features=num_features,
                num_actions=num_actions,
                non_overlap_ratio=non_overlap_ratio,
                lambda_=lambda_,
                random_state=sim * 100,
            )

            pi = eps_greedy_policy(logged_data["q"])

            estimated_policy_values = dict()
            q_hat = fit_predict_by_MLP_for_all_actions(
                features=logged_data["x_t"],
                rewards=logged_data["r"],
                num_actions=logged_data["num_actions"],
                random_state=sim * 100,
            )
            estimated_policy_values["DM"] = calc_dm(pi, q_hat)
            estimated_policy_values["IPS"] = calc_ips(logged_data, pi)
            estimated_policy_values["DR"] = calc_dr(logged_data, pi, q_hat)
            estimated_policy_values["DOLCE"] = calc_dolce(logged_data, pi)
            estimated_policy_value_list.append(estimated_policy_values)

        result_df_list.append(
            aggregate_simulation_results(
                estimated_policy_value_list,
                true_value,
                "num_actions",
                num_actions,
            )
        )

    result_df_num_actions = pd.concat(result_df_list).reset_index(level=0)
    with open("./result_df_num_actions.pkl", "wb") as f:
        pickle.dump(result_df_num_actions, f)


if __name__ == "__main__":
    main()
