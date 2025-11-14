# -*- coding: utf-8 -*-

import pandas as pd
import re

# %%
seal_path ="../data/gold/boats_itboat_seal_gold.csv"
motor_path ="../data/gold/boats_itboat_MY_gold.csv"

seal_df = pd.read_csv(seal_path)
motor_df = pd.read_csv(motor_path)

# %%

def extract_price_currency(price_str):
    if not isinstance(price_str, str):
        return None, None
    if "Price on request" in price_str:
        return None, None
    
    match = re.search(r'([\$€£])\s?([\d,]+)', price_str)
    if match:
        symbol = match.group(1)
        price = match.group(2).replace(',', '')  # Keep as string
        currency_map = {"$": "USD", "€": "EUR", "£": "GBP"}
        currency = currency_map.get(symbol, None)
        return price, currency
    return None, None

def extract_year(year_str):
    if not isinstance(year_str, str):
        return None
    match = re.search(r'(\d{4})', year_str)
    if match:
        return match.group(1)  # Return as string
    return None

# %%
motor_df["price_n"], motor_df["currency"] = zip(*motor_df["price"].map(extract_price_currency))
motor_df["year"] = motor_df["period_of_manufacture"].map(extract_year)

seal_df["price_n"], seal_df["currency"] = zip(*seal_df["price"].map(extract_price_currency))
seal_df["year"] = seal_df["period_of_manufacture"].map(extract_year)

# %%
motor_df.to_csv(motor_path, index=False)
print("Updated motor_df saved to", motor_df)

seal_df.to_csv(seal_path, index=False)
print("Updated seal_df saved to", seal_df)
