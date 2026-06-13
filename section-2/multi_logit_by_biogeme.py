import os
import pandas as pd
from IPython.core.display_functions import display

from biogeme.biogeme import BIOGEME
from biogeme.database import Database
from biogeme.expressions import Beta, Variable
from biogeme.models import loglogit
from biogeme.results_processing import get_pandas_estimated_parameters,EstimationResults

import biogeme.results_processing as res_proc

from load_data import load_data

os.chdir('./output')

def main():

    # データの読み込み（属性情報なしの場合）
    usecols = ["price_1", "price_2", "choice"]
    data_for_estimation = load_data(usecols)
    database = Database("tab_choice", data_for_estimation)

    # Define variables
    price_Kinoko = Variable("price_1")
    price_Takenoko = Variable("price_2")
    choice = Variable("choice")

    # パラメータの定義
    alpha = Beta("alpha", 0, None, None, 0)
    beta_Kinoko = Beta("beta_Kinoko", 0, None, None, 0)
    beta_Takenoko = Beta("beta_Takenoko", 0, None, None, 0)

    # 効用関数の定義
    utility_Kinoko = beta_Kinoko - alpha * price_Kinoko
    utility_Takenoko = beta_Takenoko - alpha * price_Takenoko
    utility_neither = 0

    # 各選択肢の効用関数を辞書で保存
    utilities = {0: utility_neither, 1: utility_Kinoko, 2: utility_Takenoko}

    # 選択確率の対数を計算
    log_choice_prob = loglogit(utilities, None, choice)

    # Biogemeオブジェクトの作成
    biogeme_obj = BIOGEME(database, log_choice_prob)
    biogeme_obj.model_name = "multi_logit_model"

    # null モデルの対数尤度の計算
    biogeme_obj.calculate_null_loglikelihood(avail={0: 1, 1: 1, 2: 1})

    # パラメータの推定
    results = biogeme_obj.estimate()
    
    # RawEstimationResults を EstimationResults オブジェクトに変換
    #est_results = EstimationResults(results)

    # 通常の（ロバストでない）標準誤差を含む結果を取得
    # 引数「use_robust=False」を指定します
    # pandas_results = results.get_pandas_estimated_parameters(use_robust=False)
    # print(pandas_results)

    print("推定結果の出力:")
    print(results.short_summary())

    # 推定されたパラメータをpandasに変換
    estimated_params = get_pandas_estimated_parameters(results)
    display(estimated_params)
    # estimated_params = res_proc.get_pandas_estimated_parameters(results, robust=False)
    # display(estimated_params)

if __name__ == "__main__":
    main()
