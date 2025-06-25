import pandas as pd
import arff
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("./data/processed/wtb_features.csv")
df = df.sort_values("Timestamp").reset_index(drop=True)

feature_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Prtv',
                'hour', 'minute', 'time_index', 'weekday', 'Wdir_sin', 'Wdir_cos',
                'is_operating', 'is_idle', 'is_stopped', 'x', 'y']
df = df[feature_cols + ['Patv']].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[feature_cols])
df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
df_scaled['Patv'] = df['Patv'].values  # y non scalato

attributes = [(col, 'REAL') for col in df_scaled.columns]
data = df_scaled.astype(float).values.tolist()

arff_dict = {
    'relation': 'wind_power',
    'attributes': attributes,
    'data': data
}

with open('./data/wind.arff', 'w') as f:
    arff.dump(arff_dict, f)

print("ARFF correctly saved in ./data/wind.arff")
