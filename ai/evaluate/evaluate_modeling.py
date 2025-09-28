import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import time

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

train = df.iloc[:115501]
test = df.iloc[115501:]

features_1 = ['Total Length of Fwd Packets', 'Fwd Packet Length Mean', 'Fwd Packets/s', 'Fwd IAT Mean', 'Bwd Packet Length Mean', 'Flow IAT Max', 'Fwd Packet Length Max', 'Flow IAT Std', 'Flow IAT Mean', 'Min Packet Length', 'Packet Length Variance', 'Init_Win_bytes_forward', 'Fwd Packet Length Min', 'Idle Std', 'Packet Length Mean', 'ACK Flag Count', 'Bwd Packets/s', 'Packet Length Std', 'Max Packet Length', 'Flow Packets/s', 'Average Packet Size']
features_2 = ['Init_Win_bytes_forward', 'Fwd Packet Length Mean', 'Packet Length Mean', 'ACK Flag Count', 'Bwd Packets/s', 'Total Length of Fwd Packets', 'Flow IAT Max', 'Fwd Packet Length Max', 'Packet Length Std', 'Max Packet Length', 'Flow Packets/s', 'Average Packet Size', 'Flow IAT Mean', 'Packet Length Variance']
features_3 = ['Init_Win_bytes_forward', 'Fwd Packet Length Mean', 'Packet Length Mean', 'Fwd Packet Length Max', 'Max Packet Length', 'Flow IAT Mean', 'Packet Length Variance']
features_4 = ['Fwd PSH Flags', 'Min Packet Length', 'Fwd URG Flags', 'Bwd Avg Bulk Rate', 'Flow Packets/s', 'Total Length of Fwd Packets', 'Fwd IAT Std', 'Bwd Header Length', 'Total Backward Packets', 'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Fwd Header Length', 'Fwd Avg Bulk Rate', 'Avg Bwd Segment Size', 'Init_Win_bytes_backward', 'act_data_pkt_fwd', 'Bwd IAT Max', 'Fwd Avg Bytes/Bulk', 'Flow Bytes/s', 'Fwd IAT Total', 'Bwd Avg Bytes/Bulk', 'Init_Win_bytes_forward', 'Packet Length Variance', 'URG Flag Count', 'Bwd Packets/s', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Packet Length Mean', 'Packet Length Std', 'Subflow Fwd Bytes', 'Flow IAT Std', 'Idle Max', 'SYN Flag Count', 'Down/Up Ratio', 'Fwd IAT Max', 'Bwd IAT Mean', 'Flow IAT Max', 'Total Length of Bwd Packets', 'Fwd Packet Length Max', 'Bwd Packet Length Std', 'Active Mean', 'Flow IAT Mean', 'ACK Flag Count', 'Average Packet Size', 'Flow Duration', 'Active Min', 'Idle Mean', 'Fwd IAT Mean', 'Idle Std', 'Fwd Header Length.1', 'Active Std', 'Bwd Packet Length Mean', 'PSH Flag Count', 'Fwd IAT Max, Flow Packets/s', 'Max Packet Length', 'min_seg_size_forward', 'Subflow Bwd Bytes', 'Bwd IAT Std', 'Fwd Packets/s', 'Bwd IAT Total', 'Subflow Bwd Packets', 'CWE Flag Count', 'Bwd IAT Min', 'Fwd Packet Length Min']

# X_train = train.iloc[:, 8:-3].values
X_train = train[features_3].values
y_train = train["Label"].values
# X_test = test.iloc[:, 8:-3].values
X_test = test[features_3].values
y_test = test["Label"].values

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = RobustScaler(quantile_range=(5, 95))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

window_size = 50

X_train, y_train = create_windows(X_train, y_train, window_size)
X_test, y_test = create_windows(X_test, y_test, window_size)

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

batch_size = X_test.shape[0]

# 1. Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X_train, y_train)
start_time = time.time()
nb_pred = nb_model.predict(X_test)
end_time = time.time()

accuracy = accuracy_score(y_test, nb_pred)
precision = precision_score(y_test, nb_pred)
recall = recall_score(y_test, nb_pred)
f1 = f1_score(y_test, nb_pred)
cm = confusion_matrix(y_test, nb_pred)
report = classification_report(y_test, nb_pred)
inference_time = end_time - start_time
avg_time_per_sample = inference_time / batch_size

print("------------------NB---------------")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Inference time: {inference_time:.4f} seconds')
print(f'Average inference time per sample: {avg_time_per_sample:.9f} seconds')
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)
print("----------------------------------")

# 2. Random Forest
rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=2,
    random_state=42
)
rf_model.fit(X_train, y_train)
start_time = time.time()
rf_pred = rf_model.predict(X_test)
end_time = time.time()

accuracy = accuracy_score(y_test, rf_pred)
precision = precision_score(y_test, rf_pred)
recall = recall_score(y_test, rf_pred)
f1 = f1_score(y_test, rf_pred)
cm = confusion_matrix(y_test, rf_pred)
report = classification_report(y_test, rf_pred)
inference_time = end_time - start_time
avg_time_per_sample = inference_time / batch_size

print("------------------RF---------------")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Inference time: {inference_time:.4f} seconds')
print(f'Average inference time per sample: {avg_time_per_sample:.9f} seconds')
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)
print("----------------------------------")

# 3. K-Nearest Neighbors
knn_model = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='minkowski'
)
knn_model.fit(X_train, y_train)
start_time = time.time()
knn_pred = knn_model.predict(X_test)
end_time = time.time()

accuracy = accuracy_score(y_test, knn_pred)
precision = precision_score(y_test, knn_pred)
recall = recall_score(y_test, knn_pred)
f1 = f1_score(y_test, knn_pred)
cm = confusion_matrix(y_test, knn_pred)
report = classification_report(y_test, knn_pred)
inference_time = end_time - start_time
avg_time_per_sample = inference_time / batch_size

print("------------------KNN---------------")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Inference time: {inference_time:.4f} seconds')
print(f'Average inference time per sample: {avg_time_per_sample:.9f} seconds')
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)
print("----------------------------------")

# 4. Logistic Regression
lr_model = LogisticRegression(
    penalty='l2',
    C=1.0,
    solver='lbfgs',
    max_iter=100
)
lr_model.fit(X_train, y_train)
start_time = time.time()
lr_pred = lr_model.predict(X_test)
end_time = time.time()

accuracy = accuracy_score(y_test, lr_pred)
precision = precision_score(y_test, lr_pred)
recall = recall_score(y_test, lr_pred)
f1 = f1_score(y_test, lr_pred)
cm = confusion_matrix(y_test, lr_pred)
report = classification_report(y_test, lr_pred)
inference_time = end_time - start_time
avg_time_per_sample = inference_time / batch_size

print("------------------LR---------------")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Inference time: {inference_time:.4f} seconds')
print(f'Average inference time per sample: {avg_time_per_sample:.9f} seconds')
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)
print("----------------------------------")

# 5. Support Vector Machine
svm_model = SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    max_iter=1000
)
svm_model.fit(X_train, y_train)
start_time = time.time()
svm_pred = svm_model.predict(X_test)
end_time = time.time()

accuracy = accuracy_score(y_test, svm_pred)
precision = precision_score(y_test, svm_pred)
recall = recall_score(y_test, svm_pred)
f1 = f1_score(y_test, svm_pred)
cm = confusion_matrix(y_test, svm_pred)
report = classification_report(y_test, svm_pred)
inference_time = end_time - start_time
avg_time_per_sample = inference_time / batch_size

print("------------------SVM---------------")
print(f'Accuracy: {accuracy:.4f}')
print(f'Precision: {precision:.4f}')
print(f'Recall: {recall:.4f}')
print(f'F1 Score: {f1:.4f}')
print(f'Inference time: {inference_time:.4f} seconds')
print(f'Average inference time per sample: {avg_time_per_sample:.9f} seconds')
print('Confusion Matrix:\n', cm)
print('Classification Report:\n', report)
print("----------------------------------")