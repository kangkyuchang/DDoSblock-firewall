from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
import matplotlib.pyplot as plt

def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data.iloc[i:i + window_size].values for i in range(num_samples)])
    y = np.array([
        1 if (labels.iloc[i:i + window_size] == 1).sum() > window_size / 2 else 0
        for i in range(num_samples)
    ])
    return X, y

def supervised_discretization(df, feature_col, label_col, max_depth=3):
    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(df[[feature_col]], df[label_col])

    thresholds = []
    tree = dt.tree_
    for i in range(tree.node_count):
        if tree.feature[i] == 0:
            thresholds.append(tree.threshold[i])
    thresholds = sorted(set(thresholds))

    def discretize(value):
        for i, th in enumerate(thresholds):
            if value <= th:
                return i
        return len(thresholds)

    df[feature_col] = df[feature_col].apply(discretize)
    return df

df = pd.read_csv("../DDoSdataset.csv")
df.columns = df.columns.str.strip()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

X = df.iloc[:, 8:-3]
y = df["Label"]
features = X.columns  # 이산화할 컬럼 리스트

imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=features)

scaler = RobustScaler(quantile_range=(5, 95))
X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=features)

df_scaled = pd.concat([X_scaled, y], axis=1)

for f in features:
    df_scaled = supervised_discretization(df_scaled, f, "Label", max_depth=3)

df_scaled["Label"] = df_scaled["Label"].map(lambda x: 0 if x == "BENIGN" else 1)

X = df_scaled[features]
y = df_scaled["Label"]

X, y = create_windows(X, y, 50)

X = X.reshape(X.shape[0], -1)

selector = SelectKBest(score_func=chi2, k=X.shape[1])
selector.fit_transform(X, y)

scores = selector.scores_

valid_indices = np.where(~np.isnan(scores))[0]
valid_scores = scores[valid_indices]

feature_count = len(features)

feature_indices = [i % feature_count for i in range(len(valid_scores))]

feature_aggregated = np.zeros(feature_count)
feature_counts = np.zeros(feature_count)

for idx, feat_idx in enumerate(feature_indices):
    feature_aggregated[feat_idx] += valid_scores[idx]
    feature_counts[feat_idx] += 1

feature_mean = feature_aggregated / feature_counts

feature_series = pd.Series(feature_mean, index=features)

threshold = np.mean(feature_mean)
print(f"Threshold: {threshold}")

feature_series.plot(kind="bar", figsize=(12, 6))
plt.axhline(y=float(threshold), color='red', linestyle='--', linewidth=1.5)
plt.show()

selected_features = feature_series[feature_series >= threshold]

selected_features = selected_features.sort_values(ascending=False)

print(selected_features)

# threshold = np.mean(valid_scores)

# selected_indices = [i for i, score in enumerate(scores) if score > threshold]
# selected_chi2 = set(np.where(selector.scores_ > threshold)[0])
# X_selected = X[:, selected_indices]
# print(selected_chi2)
#
# feature_names = set()
#
# feature_count = len(features)
# for i in selected_chi2:
#     feature_idx = i % feature_count
#     feature_names.add(features[feature_idx])
#
# print(len(feature_names))
