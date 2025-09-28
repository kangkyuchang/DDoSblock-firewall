from keras import models
import pandas as pd
import numpy as np
import pickle
from cicflowmeter.flow import Flow
from keras.optimizers import Adam

WINDOW_SIZE = 50

model = models.load_model("./ai/modeling/trained_model/1d_cnn_best.h5")
with open("./ai/modeling/preprocessing/imputer.pkl", "rb") as f:
    imputer = pickle.load(f)
with open("./ai/modeling/preprocessing/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)


def preprocessing(data : dict[str, Flow]) -> np.ndarray:
    COLUMNS = ["init_fwd_win_byts", "fwd_pkt_len_mean", "pkt_len_mean", "ack_flag_cnt", "bwd_pkts_s",
            "totlen_fwd_pkts", "flow_iat_max", "fwd_pkt_len_max", "pkt_len_std", "pkt_len_max", 
            "flow_pkts_s", "pkt_size_avg", "flow_iat_mean", "pkt_len_var"]
    df = pd.DataFrame(columns=COLUMNS)
    index = 0
    for key in data:
        flow = data[key]
        df.loc[index] = flow.extract_data()
        index += 1
    while index < WINDOW_SIZE:
        df.loc[index] = [0 for _ in range(14)]
        index += 1
    return df.to_numpy()

def predict(data): 
    processed_data = preprocessing(data)
    
    X = imputer.transform(processed_data)
    X = scaler.transform(X)

    X = processed_data[np.newaxis, ...] 

    single_value = model.predict(X).item() 
    result = int(single_value)

    return result

