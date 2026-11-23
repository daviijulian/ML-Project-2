import pandas as pd

DATA_DIR = "HMSdata_prepared"
VOTE_COLS = ["seizure_vote", "lpd_vote", "gpd_vote",
             "lrda_vote", "grda_vote", "other_vote"]

def load_data(include_low_votes=False):
    feats = pd.read_csv(f"{DATA_DIR}/features.csv")
    labels = pd.read_csv(f"{DATA_DIR}/label.csv")
    split = pd.read_csv(f"{DATA_DIR}/train_test.csv")[["segment_id", "group"]]
    df = feats.merge(labels.drop(columns="patient_id"), on="segment_id").merge(split, on="segment_id")

    train_groups = ["train_pool"] + (["excluded_low_votes"] if include_low_votes else [])
    train = df[df.group.isin(train_groups)]
    test = df[df.group == "test"]          # excluded_test_patient is never used
    return train, test