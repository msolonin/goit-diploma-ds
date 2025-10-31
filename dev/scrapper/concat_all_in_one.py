import pandas as pd
import re


if __name__ == '__main__':
    MY_GOLD = "data/boats_itboat_MY_gold.csv"
    PB_GOLD = "data/boats_itboat_PB_gold.csv"
    SEAL_GOLD = "data/boats_itboat_seal_gold.csv"
    GOLD_FILE_NAME = "data/boats_itboat_ALL_gold.csv"
    df1 = pd.read_csv(MY_GOLD)
    df2 = pd.read_csv(PB_GOLD)
    df3 = pd.read_csv(SEAL_GOLD)
    df_final = pd.concat([df1, df2, df3], ignore_index=True)
    df_final.to_csv(GOLD_FILE_NAME, index=False)
    print("Clean dataset saved as parsed_boat_data.csv")
