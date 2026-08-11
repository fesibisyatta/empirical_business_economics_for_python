import pandas as pd

def load_data(usecols):
    path = "/Users/shinodzukahiroshiichirou/Desktop/programing/Git/empirical_business_economics/01_Discrete_Choice_Ch02"
    data_for_estimation = pd.read_csv(
        f"{path}/output/data_for_estimation_v2.csv",
        usecols=usecols
    )
    data_for_estimation["choice"] = data_for_estimation["choice"].astype(int)  # choice列を整数型に変換
    return data_for_estimation
