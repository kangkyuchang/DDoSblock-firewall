import pickle
import numpy as np
import pandas as pd
from keras import layers, models, regularizers
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

def create_windows(data, labels, window_size):
    num_samples = data.shape[0] - window_size + 1
    X = np.array([data[i:i+window_size] for i in range(num_samples)])
    y = np.array([
        1 if np.sum(labels[i:i + window_size] == 1) > window_size / 2 else 0
        for i in range(num_samples)
    ])
    return X, y

df = pd.read_csv("./DDoSdataset.csv")
df.columns = df.columns.str.strip()

df["Timestamp"] = pd.to_datetime(df["Timestamp"])
df = df.sort_values("Timestamp").reset_index(drop=True)

df['Label'] = df['Label'].map(lambda x: 0 if x == "BENIGN" else 1)

train = df.iloc[:115501]
test = df.iloc[115501:]

features = ['Init_Win_bytes_forward', 'Fwd Packet Length Mean', 'Packet Length Mean', 'ACK Flag Count', 'Bwd Packets/s', 'Total Length of Fwd Packets', 'Flow IAT Max', 'Fwd Packet Length Max', 'Packet Length Std', 'Max Packet Length', 'Flow Packets/s', 'Average Packet Size', 'Flow IAT Mean', 'Packet Length Variance']

X_train = train[features].values
y_train = train["Label"].values
X_test = test[features].values
y_test = test["Label"].values

imputer = SimpleImputer(strategy="median")
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

scaler = RobustScaler(quantile_range=(5, 95))
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

with open('imputer.pkl', 'wb') as f:
    pickle.dump(imputer, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

window_size = 50

X_train, y_train = create_windows(X_train, y_train, window_size)
X_test, y_test = create_windows(X_test, y_test, window_size)

model = models.Sequential([
    layers.Conv1D(64, 3, activation='relu', padding="same", input_shape=(X_train.shape[1], X_train.shape[2])),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Conv1D(128, 3, activation='relu', padding="same"),
    layers.BatchNormalization(),
    layers.MaxPooling1D(2),
    layers.Dropout(0.3),

    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1),
    ModelCheckpoint('1d_cnn_best.h5', save_best_only=True)
]

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=callbacks
)