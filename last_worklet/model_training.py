import os
import numpy as np
import pandas as pd
import cupy as cp
import xgboost as xgb
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report, roc_auc_score, log_loss,
    balanced_accuracy_score, cohen_kappa_score
)
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

# ------------------- Config -------------------
dataset_base_path = "/mnt/data/saikrishna/Team_4/preprocessed_data_new"
healthy_path = os.path.join(dataset_base_path, "healthy")
mdd2_path = os.path.join(dataset_base_path, "mdd2")
mdd3_path = os.path.join(dataset_base_path, "mdd3")
file_counts = [200, 180, 160, 150, 140, 132]
random_states = list(range(1, 11)) + [42, 100, 123, 2021, 7]
output_csv = "random_state_results_mdd2_mdd3_variants.csv"

# ------------------- Load and Preprocess -------------------
def load_subset(directory, label, num_files):
    data_frames = []
    files = sorted([f for f in os.listdir(directory) if f.endswith(".csv")])[:num_files]
    for file in files:
        df = pd.read_csv(os.path.join(directory, file))
        df["Label"] = label
        match = re.search(r"(sub-\d+)", file)
        df["subject_id"] = match.group(1) if match else "unknown"
        data_frames.append(df)

    df = pd.concat(data_frames, ignore_index=True)

    for col in df.select_dtypes(include=['object']).columns:
        if col != "subject_id":
            df[col] = LabelEncoder().fit_transform(df[col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    return df

# ------------------- Main Loop -------------------
all_results = []

for count in file_counts:
    for mdd_type, mdd_path in [("mdd2", mdd2_path), ("mdd3", mdd3_path)]:
        print(f"\n=========== File Count: {count} each | Comparing healthy vs {mdd_type} ===========\n")

        df_healthy = load_subset(healthy_path, label=0, num_files=count)
        df_mdd = load_subset(mdd_path, label=1, num_files=count)
        df = pd.concat([df_healthy, df_mdd], axis=0).reset_index(drop=True)

        for state in random_states:
            print(f"\n---- Random State: {state} ----\n")
            unique_subjects = df["subject_id"].unique()
            train_ids, test_ids = train_test_split(unique_subjects, test_size=0.2, random_state=state)
            val_ids, test_ids = train_test_split(test_ids, test_size=0.5, random_state=state)

            train_df = df[df["subject_id"].isin(train_ids)]
            val_df = df[df["subject_id"].isin(val_ids)]

            X_train = train_df.drop(columns=["Label", "subject_id"]).values
            y_train = train_df["Label"].values
            X_val = val_df.drop(columns=["Label", "subject_id"]).values
            y_val = val_df["Label"].values

            scaler = MinMaxScaler(feature_range=(0, 100))
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            xgb_model = xgb.XGBClassifier(
                objective="binary:logistic",
                tree_method="hist",
                device="cuda",
                eval_metric="logloss",
                learning_rate=0.005,
                max_depth=25,
                gamma=0.2,
                subsample=0.95,
                colsample_bytree=0.97,
                min_child_weight=1,
                reg_alpha=0.8,
                reg_lambda=3.0,
                n_estimators=6000,
                verbosity=0
            )

            calibrated_model = CalibratedClassifierCV(estimator=xgb_model, method='sigmoid', cv=3)
            calibrated_model.fit(X_train_scaled, y_train)

            xgb_model.fit(X_train_scaled, y_train)

            X_val_gpu = cp.array(X_val_scaled)
            val_preds = xgb_model.predict(X_val_gpu)
            val_probs = xgb_model.predict_proba(X_val_gpu)[:, 1]

            accuracy = accuracy_score(y_val, val_preds)
            roc_auc = roc_auc_score(y_val, val_probs)
            logloss_val = log_loss(y_val, val_probs)
            balanced_acc = balanced_accuracy_score(y_val, val_preds)
            kappa = cohen_kappa_score(y_val, val_preds)

            all_results.append({
                "dataset": mdd_type,
                "file_count": count,
                "random_state": state,
                "accuracy": accuracy,
                "balanced_accuracy": balanced_acc,
                "kappa": kappa,
                "log_loss": logloss_val,
                "roc_auc": roc_auc
            })

# ------------------- Save All Results -------------------
results_df = pd.DataFrame(all_results)
results_df.to_csv(output_csv, index=False)
print(f"\nAll metrics saved to {output_csv}")
