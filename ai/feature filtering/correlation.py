import pandas as pd
from scipy import stats

df = pd.read_csv("../DDoSdataset.csv")
df.columns = df.columns.str.strip()

# features_1 = {'Subflow Fwd Bytes', 'Fwd Packet Length Mean', 'Fwd Packets/s', 'Fwd IAT Mean', 'Bwd Packet Length Mean', 'Flow IAT Max', 'Fwd Packet Length Max', 'Flow IAT Std', 'Flow IAT Mean', 'Min Packet Length', 'Packet Length Variance', 'Init_Win_bytes_forward', 'Fwd Packet Length Min', 'Idle Std', 'Avg Fwd Segment Size', 'Packet Length Mean', 'ACK Flag Count', 'Bwd Packets/s', 'Total Length of Fwd Packets', 'Packet Length Std', 'Max Packet Length', 'Flow Packets/s', 'Average Packet Size'}
# features_2 = ['Subflow Fwd Bytes', 'Init_Win_bytes_forward', 'Fwd Packet Length Mean', 'Avg Fwd Segment Size', 'Packet Length Mean', 'ACK Flag Count', 'Bwd Packets/s', 'Total Length of Fwd Packets', 'Flow IAT Max', 'Fwd Packet Length Max', 'Packet Length Std', 'Max Packet Length', 'Flow Packets/s', 'Average Packet Size', 'Flow IAT Mean', 'Packet Length Variance']
features_3 = ['Init_Win_bytes_forward', 'Fwd Packet Length Mean', 'Packet Length Mean', 'Fwd Packet Length Max', 'Max Packet Length', 'Flow IAT Mean', 'Packet Length Variance']

# ordered_features_1 = [col for col in df.columns if col in features_1]
# ordered_features_2 = [col for col in df.columns if col in features_2]
ordered_features_3 = [col for col in df.columns if col in features_3]

# df_features_1 = df[ordered_features_1]
# df_features_2 = df[ordered_features_2]
df_features_3 = df[ordered_features_3]

pearson = []
pearson_P_value = []

spearman = []
spearman_P_value = []

kendall = []
kendall_P_value = []

for i in range(0, len(df_features_3.columns)):
    arr1 = []
    p_values1 = []
    arr2 = []
    p_values2 = []
    arr3 = []
    p_values3 = []
    for j in range(0, len(df_features_3.columns)):
        x = df_features_3[df_features_3.columns[i]]
        y = df_features_3[df_features_3.columns[j]]
        pearson_corr, p_value1 = stats.pearsonr(x, y)
        spearman_corr, p_value2 = stats.spearmanr(x, y)
        kendall_corr, p_value3 = stats.kendalltau(x, y)
        arr1.append(pearson_corr)
        p_values1.append(p_value1)
        arr2.append(spearman_corr)
        p_values2.append(p_value2)
        arr3.append(kendall_corr)
        p_values3.append(p_value3)
    pearson.append(arr1)
    pearson_P_value.append(p_values1)
    spearman.append(arr2)
    spearman_P_value.append(p_values2)
    kendall.append(arr3)
    kendall_P_value.append(p_values3)

df_features_pearson = pd.DataFrame(pearson, index=df_features_3.columns, columns=df_features_3.columns)
df_features_pearson_p = pd.DataFrame(pearson_P_value, index=df_features_3.columns, columns=df_features_3.columns)
df_features_spearman = pd.DataFrame(spearman, index=df_features_3.columns, columns=df_features_3.columns)
df_features_spearman_p = pd.DataFrame(spearman_P_value, index=df_features_3.columns, columns=df_features_3.columns)
df_features_kendall = pd.DataFrame(kendall, index=df_features_3.columns, columns=df_features_3.columns)
df_features_kendall_p = pd.DataFrame(kendall_P_value, index=df_features_3.columns, columns=df_features_3.columns)

df_features_pearson.to_excel('pearson_3.xlsx', index=True)
df_features_pearson_p.to_excel('pearson_3_p_value.xlsx', index=True)
df_features_spearman.to_excel('spearman_3.xlsx', index=True)
df_features_spearman_p.to_excel('spearman_3_p_value.xlsx', index=True)
df_features_kendall.to_excel('kendall_3.xlsx', index=True)
df_features_kendall_p.to_excel('kendall_3_p_value.xlsx', index=True)