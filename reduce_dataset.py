import pandas as pd

df = pd.read_csv("data/raw/wtbdata_245days.csv")

turbine_ids = df["TurbID"].unique()
print(f"Total turbines: {len(turbine_ids)}")

selected_turbines = turbine_ids[:10]  # np.random.choice(turbine_ids, 10, replace=False)

df_reduced = df[df["TurbID"].isin(selected_turbines)]

df_reduced.to_csv("data/raw/wtbdata_10turbines.csv", index=False)

print(f"Dataset ridotto salvato: {df_reduced.shape[0]} rows, {len(selected_turbines)} turbines")
