import os
import pandas as pd
import matplotlib.pyplot as plt

from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable, Draws, MonteCarlo, log
from biogeme.models import logit
from biogeme.results_processing import get_pandas_estimated_parameters

from biogeme.expressions import PanelLikelihoodTrajectory

from load_data import load_data


def main():
    """混合ロジットによるパラメータ推定
    """
    
    # データの読み込み（属性情報なしの場合）
    usecols = ["ID", "price_1", "price_2", "choice"]
    data_for_estimation = load_data(usecols)
    database = Database("tab_choice", data_for_estimation)

    # パネルデータなので, 個人識別番号を付与
    database.panel("ID")

    # 変数の定義
    price_Kinoko = Variable("price_1")
    price_Takenoko = Variable("price_2")
    choice = Variable("choice")


    # パラメータの初期値
    init_params = {
        "beta_price": 0.6,
        "beta_Kinoko": 20,
        "beta_Takenoko": 20,
        "sigma_price": 0.1,
        "sigma_Kinoko": 1,
        "sigma_Takenoko": 1,
    }
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
    utilities = {0:utility_neither, 1: utility_Kinoko, 2: utility_Takenoko}

    # 条件付き確率の計算
    conditional_prob = logit(utilities,{0:1,1:1,2:1},choice)
    trajectory = PanelLikelihoodTrajectory(conditional_prob)
    logprob = log(MonteCarlo(trajectory))

    # Biogemeオブジェクトの作成
    biogeme_obj = BIOGEME(database, logprob, number_of_draws = 1000, seed = 31)
    biogeme_obj.model_name = "mixed_logit_model"

    # nullモデルの対数尤度の計算
    biogeme_obj.calculate_null_loglikelihood(avail={0: 1, 1: 1, 2: 1})

    # パラメータの推定
    results = biogeme_obj.estimate()
    
    print("推定結果の出力:")
    print(results.short_summary())

    # 推定されたパラメータをpandasに変換
    estimated_params = get_pandas_estimated_parameters(results)

    estimated_params = estimated_params.set_index("Name").reindex(
        [
            "beta_price",
            "beta_Kinoko",
            "beta_Takenoko",
            "sigma_price",
            "sigma_Kinoko",
            "sigma_Takenoko"
        ]
    )
    display(estimated_params)

if __name__ == "__main__":
    main()