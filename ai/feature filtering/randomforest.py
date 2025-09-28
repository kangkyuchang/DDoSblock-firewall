from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt


def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([
        1 if np.sum(labels[i:i + window_size] == 1) > window_size / 2 else 0
        for i in range(num_samples)
    ])
    return X, y

df = pd.read_csv("../DDoSdataset.csv")
df.columns = df.columns.str.strip()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

df["Label"] = df["Label"].map(lambda x: 0 if x == "BENIGN" else 1)

original_features = df.columns[8:-3]

X = df.iloc[:, 8:-3].values
y = df["Label"].values

imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X)

scaler = RobustScaler(quantile_range=(5, 95))
X = scaler.fit_transform(X)

window_size = 50

X, y = create_windows(X, y, window_size)

X = X.reshape(X.shape[0], -1)

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

rf.fit(X, y)

importances = rf.feature_importances_

window_length = window_size
feature_count = len(original_features)

feature_indices = [i % feature_count for i in range(len(importances))]

feature_importance_aggregated = np.zeros(feature_count)
feature_importance_counts = np.zeros(feature_count)

for idx, feat_idx in enumerate(feature_indices):
    feature_importance_aggregated[feat_idx] += importances[idx]
    feature_importance_counts[feat_idx] += 1

feature_importance_mean = feature_importance_aggregated / feature_importance_counts

feature_importance_series = pd.Series(feature_importance_mean, index=original_features)

feature_importance_series.plot(kind="bar", figsize=(12, 6))
plt.axhline(y=0.0001, color='red', linestyle='--', linewidth=1.5)
y_ticks = np.arange(0, 0.005, 0.0001)
plt.yticks(y_ticks)
plt.show()

selected_features = feature_importance_series[feature_importance_series > 0.0001]

selected_features = selected_features.sort_values(ascending=False)

print(selected_features)
