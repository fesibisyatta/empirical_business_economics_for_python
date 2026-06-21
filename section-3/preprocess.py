import numpy as np
import pandas as pd

if __name__ == "__main__":

    # GitHub raw URLs
    url_data = (
        "https://raw.githubusercontent.com/keisemi/empirical_business_economics/"
        "main/02_BLP_Ch03_04_05/data/CleanData_20180222_nippyo.csv"
    )

    url_hh = (
        "https://raw.githubusercontent.com/keisemi/empirical_business_economics/"
        "main/02_BLP_Ch03_04_05/data/HHsize.csv"
    )

    url_cpi = (
        "https://raw.githubusercontent.com/keisemi/empirical_business_economics/"
        "main/02_BLP_Ch03_04_05/data/zni2015s.csv"
    )

    # 自動車データ
    data = pd.read_csv(url_data)

    # 家計数（潜在的な市場規模）データ
    dataHH = pd.read_csv(url_hh, thousands=",")

    # 消費者物価指数
    dataCPI = pd.read_csv(url_cpi, encoding="shift_jis")

    # Rの dataCPI[6:56, ] に対応
    # Rは1始まりなので、Pythonでは iloc[5:56]
    dataCPI = dataCPI.iloc[5:56, :].copy()

    dataCPI = (
        dataCPI
        .rename(columns={
            "類・品目": "year",
            "総合": "cpi",
        })
        [["year", "cpi"]]
    )

    dataCPI["year"] = pd.to_numeric(dataCPI["year"], errors="coerce")
    dataCPI["cpi"] = pd.to_numeric(dataCPI["cpi"], errors="coerce")

    # データクリーニング ----
    # 必要な変数のみをキープ
    data = data[
        [
            "Maker", "Type", "Name", "Year", "Sales", "Model",
            "Nippyo", "price", "kata",
            "weight", "capacity", "FuelType", "FuelEfficiency", "HorsePower",
            "overall_length", "overall_width", "overall_height",
        ]
    ].copy()

    data = data.rename(columns={"Year": "year"})

    # 家計サイズをマージする
    data = data.merge(dataHH, on="year", how="left")

    # CPIをマージする
    data = data.merge(dataCPI, on="year", how="left")

    # 燃費が欠損しているデータを落とす
    data = data[data["FuelEfficiency"].notna()].copy()

    # 価格の実質化を行う。ここでは、2016年を基準年とする。
    # また、価格の単位を100万円にする。元のデータは1万円
    cpi2016 = dataCPI.loc[dataCPI["year"] == 2016, "cpi"].iloc[0]

    data["price"] = data["price"] / (data["cpi"] / cpi2016)
    data["price"] = data["price"] / 100

    data = data.drop(columns=["cpi"])

    # サイズ（高さ × 幅 × 長さ）、馬力重量比を定義する
    data["size"] = (
        (data["overall_length"] / 1000)
        * (data["overall_width"] / 1000)
        * (data["overall_height"] / 1000)
    )

    data["hppw"] = data["HorsePower"] / data["weight"]

    data = data.drop(columns=["HorsePower", "weight"])

    # starts_with("overall") に対応
    overall_cols = [col for col in data.columns if col.startswith("overall")]
    data = data.drop(columns=overall_cols)

    # 自動車の車種IDを作成する
    # dplyr::cur_group_id() は1始まりなので +1 する
    data["NameID"] = pd.factorize(data["Name"], sort=True)[0] + 1

    # relocate(NameID, year) に対応
    front_cols = ["NameID", "year"]
    other_cols = [col for col in data.columns if col not in front_cols]
    data = data[front_cols + other_cols]

    # マーケットシェアと Outside option share を定義する
    data["inside_total"] = data.groupby("year")["Sales"].transform("sum")
    data["outside_total"] = data["HH"] - data["inside_total"]
    data["share"] = data["Sales"] / data["HH"]
    data["share0"] = data["outside_total"] / data["HH"]
    data = data.drop(columns=["inside_total", "outside_total"])

    cols = ["hppw", "FuelEfficiency", "size"]
    group_cols = ["year", "Maker"]

    # sum_own
    for col in cols:
        data[f"{col}_sum_own"] = data.groupby(group_cols)[col].transform("sum")

    # sqr_sum_own
    for col in cols:
        data[f"{col}_sqr_sum_own"] = data.assign(
            _squared=data[col] ** 2
        ).groupby(group_cols)["_squared"].transform("sum")

    # group_n
    data["group_n"] = data.groupby(group_cols)["year"].transform("size")

    # sum_own
    group_col=["year"]
    for col in cols:
        data[f"{col}_sum_mkt"] = data.groupby(group_col)[col].transform("sum")

    # sqr_sum_own
    for col in cols:
        data[f"{col}_sqr_sum_mkt"] = data.assign(
            _squared=data[col] ** 2
        ).groupby(group_col)["_squared"].transform("sum")

    # mkt_n
    data["mkt_n"] = data.groupby(group_col)["year"].transform("size")

    # BLP操作変数の定義
    for col in cols:
        data[f"iv_BLP_own_{col}"] = data[f"{col}_sum_own"] - data[f"{col}"]
        data[f"iv_BLP_other_{col}"] = data[f"{col}_sum_mkt"] - data[f"{col}_sum_own"]
    
    # 差別化操作変数の定義
    for col in cols:
        # 同じ企業間の商品距離
        data[f"iv_GH_own_{col}"] = (
            (data["group_n"] - 1)*data[f"{col}"]**2 
            - 2 * data[f"{col}"] * (data[f"{col}_sum_own"] - data[f"{col}"]) 
            + (data[f"{col}_sqr_sum_own"] - data[f"{col}"]**2)
            )

        # ライバル企業との製品間の距離
        data[f"iv_GH_other_{col}"] = (
            (data["mkt_n"] - data["group_n"]) * (data[f"{col}"]**2) 
            + (data[f"{col}_sqr_sum_mkt"] - data[f"{col}_sqr_sum_own"]) 
            - 2 * data[f"{col}"] * (data[f"{col}_sum_mkt"] - data[f"{col}_sum_own"])
        )
    
    # 被説明変数
    data["logit_share"] = np.log(data["share"] / data["share0"])

    data.to_csv("output/processed_data.csv", index=False)