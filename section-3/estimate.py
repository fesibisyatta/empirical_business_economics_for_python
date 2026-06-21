import numpy as np
import pandas as pd
import statsmodels.api as sm
from linearmodels.iv import IV2SLS

if __name__ == "__main__":

    data = pd.read_csv("output/processed_data.csv")

    # OLSの推定
    print("OLS estimation:")
    X = data[["price", "hppw", "FuelEfficiency", "size"]]
    X = sm.add_constant(X)
    y = data["logit_share"]
    model = sm.OLS(y, X).fit(cov_type="HC1")
    print(model.summary())

    print(" ")
    print(" ")
    print(" ")

    # BLP操作変数を用いたIV推定
    print("\nIV estimation with BLP instruments:")
    formula = """
    logit_share ~ 1 + hppw + FuelEfficiency + size
    + [price ~ iv_BLP_own_hppw
            + iv_BLP_own_FuelEfficiency
            + iv_BLP_own_size
            + iv_BLP_other_hppw
            + iv_BLP_other_FuelEfficiency
            + iv_BLP_other_size]
    """
    
    iv_BLP = IV2SLS.from_formula(
        formula,
        data=data
    ).fit(cov_type="robust")

    print(iv_BLP.summary)

    print(" ")
    print(" ")
    print(" ")

    # 差別化操作変数を用いたIV推定
    print("\nIV estimation with differentiation instruments:")
    formula = """
    logit_share ~ 1 + hppw + FuelEfficiency + size
    + [price ~ iv_GH_own_hppw
            + iv_GH_own_FuelEfficiency
            + iv_GH_own_size
            + iv_GH_other_hppw
            + iv_GH_other_FuelEfficiency
            + iv_GH_other_size]
    """

    iv_GH = IV2SLS.from_formula(
        formula,
        data=data
    ).fit(cov_type="heteroskedastic")

    print(iv_GH.summary)