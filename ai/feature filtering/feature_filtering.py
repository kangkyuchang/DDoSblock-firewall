import pandas as pd

# df = pd.read_csv("../DDoSdataset.csv")
# df.columns = df.columns.str.strip()
#
# df["Timestamp"] = pd.to_datetime(df["Timestamp"])
# df = df.sort_values("Timestamp").reset_index(drop=True)
#
# df["Label"] = df["Label"].map(lambda x: 0 if x == "BENIGN" else 1)

information_gain_features = {"Average Packet Size", "Packet Length Mean", "Fwd Packet Length Mean",
                            "Total Length of Fwd Packets", "Avg Fwd Segment Size", "Subflow Fwd Bytes", "Flow Bytes/s",
                            "Max Packet Length", "Flow IAT Max", "Flow Packets/s", "Fwd Packet Length Max",
                            "Flow Duration", "Fwd Packet Length Min", "Flow IAT Mean", "Min Packet Length",
                            "Fwd Packets/s", "Fwd IAT Max", "Fwd IAT Total", "Flow IAT Std", "Fwd IAT Mean",
                            "Bwd Packets/s", "Packet Length Std", "Packet Length Variance", "Init_Win_bytes_forward",
                            "Fwd Header Length.1", "Fwd Header Length", "Bwd IAT Max", "Bwd IAT Total", "Bwd IAT Mean",
                            "Bwd Header Length", "Subflow Bwd Packets", "Total Backward Packets", "Subflow Bwd Bytes",
                            "Total Length of Bwd Packets", "Bwd Packet Length Mean", "Avg Bwd Segment Size",
                            "min_seg_size_forward", "Bwd Packet Length Max", "Bwd IAT Min"}

chai_square_features = {"Bwd Header Length", "Idle Std", "Init_Win_bytes_forward", "Bwd Packet Length Max",
                       "Subflow Fwd Bytes", "Fwd Header Length", "Max Packet Length", "Idle Max", "Total Backward Packets",
                       "Fwd Avg Bytes/Bulk", "act_data_pkt_fwd", "Flow Duration", "Avg Bwd Segment Size", "Bwd IAT Total",
                       "ACK Flag Count", "Bwd Packet Length Min", "Fwd IAT Std", "Active Min", "Bwd Packet Length Std",
                       "Bwd IAT Std", "Fwd Packet Length Max", "SYN Flag Count", "Fwd URG Flags", "Flow IAT Max",
                       "Fwd IAT Max, Flow Packets/s", "Down/Up Ratio", "Bwd Packets/s", "Active Std", "Bwd Avg Bulk Rate",
                       "Packet Length Variance", "Fwd Packet Length Mean", "Bwd Avg Bytes/Bulk", "Fwd PSH Flags",
                       "Fwd IAT Total", "CWE Flag Count", "PSH Flag Count", "Average Packet Size", "Packet Length Mean",
                       "Subflow Bwd Packets", "Fwd Avg Bulk Rate", "Active Mean", "Total Length of Fwd Packets",
                       "Total Length of Bwd Packets", "Bwd IAT Max", "Flow IAT Mean"}

random_forest_features = {"Min Packet Length", "Fwd Packet Length Min", "ACK Flag Count", "Packet Length Variance",
                          "Avg Fwd Segment Size", "Packet Length Std", "URG Flag Count", "Fwd Packet Length Mean",
                          "Average Packet Size", "Bwd Packets/s", "Init_Win_bytes_forward", "Packet Length Mean",
                          "Flow IAT Mean", "Max Packet Length", "Flow Packets/s", "Subflow Fwd Bytes",
                          "Total Length of Fwd Packets", "Fwd Packet Length Max"}

PFI_features = {"ACK Flag Count", "Packet Length Variance", "Init_Win_bytes_forward", "Avg Fwd Segment Size",
                "Fwd Packet Length Max", "Init_Win_bytes_backward", "Flow IAT Mean", "Max Packet Length",
                "Fwd Packet Length Mean", "URG Flag Count", "Packet Length Mean", "Flow Packets/s", "Fwd Packets/s",
                "Packet Length Std", "Flow IAT Std", "Fwd IAT Mean", "Idle Std", "Idle Mean", "Bwd Packet Length Mean",
                "Flow IAT Max"}

union = information_gain_features | chai_square_features | random_forest_features | PFI_features

print(union)
print(len(union))

union = dict.fromkeys(union, 0)

for feature in information_gain_features:
    union[feature] += 1

for feature in chai_square_features:
    union[feature] += 1

for feature in random_forest_features:
    union[feature] += 1

for feature in PFI_features:
    union[feature] += 1

features_dup2 = {key: 0 for key, val in union.items() if val == 2}
features_dup3 = {key for key, val in union.items() if val >= 3}
features_dup4 = {key for key, val in union.items() if val >= 4}

for key in features_dup2:
    if key in information_gain_features:
        features_dup2[key] += 1
    if key in chai_square_features:
        features_dup2[key] += 1
    if key in random_forest_features:
        features_dup2[key] += 2
    if key in PFI_features:
        features_dup2[key] += 2

features_dup2 = {key for key, val in features_dup2.items() if val % 2 != 0} | features_dup3

# print(features_dup2)
# print(features_dup3)
# print(features_dup4)


