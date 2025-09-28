from sklearn.feature_selection import mutual_info_classif
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

df['Label'] = df['Label'].map(lambda x: 0 if x == "BENIGN" else 1)

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

mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)

feature_count = len(original_features)

feature_indices = [i % feature_count for i in range(len(mi))]

feature_aggregated = np.zeros(feature_count)
feature_counts = np.zeros(feature_count)

for idx, feat_idx in enumerate(feature_indices):
    feature_aggregated[feat_idx] += mi[idx]
    feature_counts[feat_idx] += 1

feature_mean = feature_aggregated / feature_counts

feature_series = pd.Series(feature_mean, index=original_features)

threshold = np.mean(feature_mean)
print(f"Threshold: {threshold}")

feature_series.plot(kind="bar", figsize=(12, 6))
plt.axhline(y=float(threshold), color='red', linestyle='--', linewidth=1.5)
plt.show()

selected_features = feature_series[feature_series >= threshold]

selected_features = selected_features.sort_values(ascending=False)

print(selected_features)