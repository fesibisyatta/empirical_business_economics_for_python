import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log
from biogeme.models import logit
from biogeme.results_processing import get_pandas_estimated_parameters

from biogeme.expressions import PanelLikelihoodTrajectory

from load_data import load_data

os.chdir('./output')
plt.style.use('ggplot')


def create_logprob(init_params):
    """初期値を受け取り, 混合ロジットの対数尤度を作成する
    """

    # 変数の定義
    price_Kinoko = Variable("price_1")
    price_Takenoko = Variable("price_2")
    choice = Variable("choice")

    # パラメータの初期値
    beta_price_init = init_params["beta_price"]
    beta_Kinoko_init = init_params["beta_Kinoko"]
    beta_Takenoko_init = init_params["beta_Takenoko"]
    sigma_Kinoko_init = init_params["sigma_Kinoko"]
    sigma_Takenoko_init = init_params["sigma_Takenoko"]
    sigma_price_init = init_params["sigma_price"]

    # パラメータの定義
    beta_price = Beta("beta_price", beta_price_init, None, None, 0)
    beta_Kinoko = Beta("beta_Kinoko", beta_Kinoko_init, None, None, 0)
    beta_Takenoko = Beta("beta_Takenoko", beta_Takenoko_init, None, None, 0)

    # ランダム係数の標準偏差の定義
    sigma_Kinoko = Beta("sigma_Kinoko", sigma_Kinoko_init, 0, None, 0)
    sigma_Takenoko = Beta("sigma_Takenoko", sigma_Takenoko_init, 0, None, 0)
    sigma_price = Beta("sigma_price", sigma_price_init, 0, None, 0)

    # ランダム係数の定義
    beta_Kinoko_random = (
        beta_Kinoko + sigma_Kinoko * Draws("beta_Kinoko_random", "NORMAL")
    )
    beta_Takenoko_random = (
        beta_Takenoko + sigma_Takenoko * Draws("beta_Takenoko_random", "NORMAL")
    )
    beta_price_random = (
        beta_price + sigma_price * Draws("beta_price_random", "NORMAL")
    )

    # 効用関数の定義
    utility_Kinoko = beta_Kinoko_random - beta_price_random * price_Kinoko
    utility_Takenoko = beta_Takenoko_random - beta_price_random * price_Takenoko
    utility_neither = 0

    # 各選択肢の効用関数を辞書で保存
    utilities = {0: utility_neither, 1: utility_Kinoko, 2: utility_Takenoko}

    # 条件付き確率の計算
    conditional_prob = logit(utilities, {0: 1, 1: 1, 2: 1}, choice)
    trajectory = PanelLikelihoodTrajectory(conditional_prob)
    logprob = log(MonteCarlo(trajectory))

    return logprob



def main():
    """混合ロジットによるパラメータ推定
    """

    # データの読み込み（属性情報なしの場合）
    usecols = ["ID", "price_1", "price_2", "choice"]
    data_for_estimation = load_data(usecols)
    database = Database("tab_choice", data_for_estimation)

    # パネルデータなので, 個人識別番号を付与
    database.panel("ID")

    # 初期値を30通り作成
    rng = np.random.default_rng(24)

    init_params_list = []
    for init_id in range(1, 31):
        init_params = {
            "beta_price": rng.uniform(0.1, 1.2),
            "beta_Kinoko": rng.uniform(5, 50),
            "beta_Takenoko": rng.uniform(5, 50),
            "sigma_price": rng.uniform(0.01, 0.5),
            "sigma_Kinoko": rng.uniform(0.01, 5),
            "sigma_Takenoko": rng.uniform(0.01, 5),
        }

        init_params["init_id"] = init_id
        init_params_list.append(init_params)

    results_list = []

    for init_params in tqdm(init_params_list):

        init_id = init_params["init_id"]

        # init_idはBiogemeのパラメータではないので除外
        model_init_params = {
            key: value
            for key, value in init_params.items()
            if key != "init_id"
        }

        # 初期値ごとにlogprobを作り直す
        logprob = create_logprob(model_init_params)

        biogeme_obj = BIOGEME(
            database,
            logprob,
            number_of_draws=1000,
            seed=24,
            generate_html=False,
            generate_yaml=False
        )

        biogeme_obj.model_name = f"mixed_logit_init_{init_id}"

        biogeme_obj.calculate_null_loglikelihood(
            avail={0: 1, 1: 1, 2: 1}
        )

        results = biogeme_obj.estimate()
        estimated_params = get_pandas_estimated_parameters(results)

        beta_kinoko = estimated_params.loc[
            estimated_params["Name"] == "beta_Kinoko",
            "Value"
        ].iloc[0]

        final_log_likelihood = results.final_log_likelihood

        results_list.append(
            {
                "init_id": init_id,

                # 初期値
                "beta_price_init": model_init_params["beta_price"],
                "beta_Kinoko_init": model_init_params["beta_Kinoko"],
                "beta_Takenoko_init": model_init_params["beta_Takenoko"],
                "sigma_price_init": model_init_params["sigma_price"],
                "sigma_Kinoko_init": model_init_params["sigma_Kinoko"],
                "sigma_Takenoko_init": model_init_params["sigma_Takenoko"],

                # 推定後のbeta_Kinoko
                "beta_Kinoko": beta_kinoko,

                # 念のため対数尤度も保存
                "final_log_likelihood": final_log_likelihood,
            }
        )

    df = pd.DataFrame(results_list)

    display(df)

    df.to_csv("beta_Kinoko_by_all_initial_values.csv", index=False)

    # beta_Kinokoの推定値の分布
    mean_beta = df["beta_Kinoko"].mean()

    plt.figure(figsize=(8, 5))

    plt.hist(
        df["beta_Kinoko"],
        bins=10
    )

    plt.axvline(
        mean_beta,
        linestyle="--",
        label=f"Mean = {mean_beta:.4f}"
    )

    plt.xlabel("Estimated beta_Kinoko")
    plt.ylabel("Frequency")
    plt.title("Distribution of beta_Kinoko by Initial Values")
    plt.legend()
    plt.savefig("beta_Kinoko_distribution_by_all_initial_values.png")
    plt.show()

    # 各初期値とbeta_Kinokoの推定値の関係
    init_columns = [
        "beta_price_init",
        "beta_Kinoko_init",
        "beta_Takenoko_init",
        "sigma_price_init",
        "sigma_Kinoko_init",
        "sigma_Takenoko_init",
    ]

    for col in init_columns:
        plt.figure(figsize=(8, 5))

        plt.scatter(
            df[col],
            df["beta_Kinoko"]
        )

        plt.axhline(
            mean_beta,
            linestyle="--",
            label=f"Mean = {mean_beta:.4f}"
        )

        plt.xlabel(col)
        plt.ylabel("Estimated beta_Kinoko")
        plt.title(f"Estimated beta_Kinoko by {col}")
        plt.legend()
        plt.savefig(f"beta_Kinoko_by_{col}.png")
        plt.show()


if __name__ == "__main__":
    main()