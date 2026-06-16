import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import json
import os
import matplotlib.pyplot as plt


def train_different_models_and_gridsearch_best_and_validate(path_to_input_data, path_to_output_metrics):
    print("\n--- step3: train_different_models_and_gridsearch_best_and_validate ---\n")

    df = pd.read_parquet(path_to_input_data)

    df["target"] = df["was_returned"]

    train, tmp = train_test_split(df, test_size=0.4, random_state=0, stratify=df["target"])
    val, test = train_test_split(tmp, test_size=0.5, random_state=0, stratify=tmp["target"])

    del tmp
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    test = test.reset_index(drop=True)

    print("train:", train.shape, "return rate:", train["target"].mean())
    print("val:", val.shape, "return rate:", val["target"].mean())
    print("test:", test.shape, "return rate:", test["target"].mean())

    train["customerID_mean_return"] = train.groupby("customerID")["target"].transform("mean")
    train["articleID_mean_return"] = train.groupby("articleID")["target"].transform("mean")

    u2t = train.groupby("customerID")["target"].mean().to_dict()
    i2t = train.groupby("articleID")["target"].mean().to_dict()

    val["customerID_mean_return"] = val["customerID"].map(u2t)
    val["articleID_mean_return"] = val["articleID"].map(i2t)
    test["customerID_mean_return"] = test["customerID"].map(u2t)
    test["articleID_mean_return"] = test["articleID"].map(i2t)

    feature_columns = [
        "customerID_mean_return", # float
        "articleID_mean_return", # float
        "colorCode", # str
        "sizeCode", # str
        "productGroup", # str
        "price_bin",  # str
        "discount_bin", # str
    ]

    x_train = train[feature_columns]
    y_train = train["target"]
    x_val = val[feature_columns]
    y_val = val["target"]
    x_test = test[feature_columns]
    y_test = test["target"]

    # !!! не используем scaler-ы !!! единственные численные фичи - средний таргет, а он, в свою очередь, 0 или 1.
    
    models_val_results = []
    # possible_models = ["constant_answer", "KNN", "logreg", "random_forest", "nn_classifier"]

    models_val_results.append({
        "model": "constant_answer",
        "params": {},
        "val_roc_auc": roc_auc_score(y_val, [1 if train.target.mean() > 0.5 else 0] * len(y_val)),
    })

    for column in ["colorCode", "sizeCode", "productGroup", "price_bin", "discount_bin"]:
        le = LabelEncoder()
        le.fit(train[column])  
        x_train[column] = le.transform(x_train[column])
        x_val[column] = le.transform(x_val[column])
        x_test[column] = le.transform(x_test[column])

    # убедимся что в вал и тесте нет новых категор значений
    assert x_val.isna().sum().sum() == 0
    assert x_test.isna().sum().sum() == 0

    for k in [1, 5, 10, 20]:
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_val)[:, 1]
        models_val_results.append({
            "model": "KNN",
            "params": {"n_neighbors": k},
            "val_roc_auc": roc_auc_score(y_val, pred),
        })

    for c in [0.1, 1, 10, np.inf]:
    # for c in [0.1, 1]:
        model = LogisticRegression(C=c, max_iter=100)
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_val)[:, 1]
        models_val_results.append({
            "model": "LogisticRegression",
            "params": {"C": c},
            "val_roc_auc": roc_auc_score(y_val, pred)
        })

    for depth in [2, 8, 10, 12]:
    # for depth in [2,3]:
        model = RandomForestClassifier(n_estimators=30, max_depth=depth)
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_val)[:, 1]
        models_val_results.append({
            "model": "RandomForestClassifier",
            "params": {"max_depth": depth},
            "val_roc_auc": roc_auc_score(y_val, pred)
        })

    for layers in [(100,), (50,50), (33,33,33), (25,25,25,25)]:
    # for layers in [(10,), (10,10)]:
        model = MLPClassifier(hidden_layer_sizes=layers)
        model.fit(x_train, y_train)
        pred = model.predict_proba(x_val)[:, 1]
        models_val_results.append({
            "model": "MLPClassifier",
            "params": {"hidden_layer_sizes": layers},
            "val_roc_auc": roc_auc_score(y_val, pred)
        })

    for x in models_val_results:
        print(x)

    best_models_config = {"constant_answer": {}}
    for model in ["KNN", "LogisticRegression", "RandomForestClassifier", "MLPClassifier"]:
        all_results = [x for x in models_val_results if x["model"] == model]
        all_results = sorted(all_results, key=lambda x: x["val_roc_auc"])
        best_models_config[model] = all_results[-1]["params"]

    print("нашли лучшие модели гридсёрчем по отложенной выборке")
    for k, v in best_models_config.items():
        print(k, " : ", v)


    print("\nтеперь обучаем на совмещенном трейн+вал и дальше уже получаем итоговые результаты моделей на тесте и сравниваем")

    x_train_full = pd.concat([x_train, x_val]).reset_index(drop=True)
    y_train_full = pd.concat([y_train, y_val]).reset_index(drop=True)

    const_test_metrics = {
        "accuracy": accuracy_score(y_test, [1 if y_train_full.mean() > 0.5 else 0] * len(y_test)),
        "precision": precision_score(y_test, [1 if y_train_full.mean() > 0.5 else 0] * len(y_test)),
        "recall": recall_score(y_test, [1 if y_train_full.mean() > 0.5 else 0] * len(y_test)),
        "f1_score": f1_score(y_test, [1 if y_train_full.mean() > 0.5 else 0] * len(y_test)),
        "roc_auc_score": roc_auc_score(y_test, [1 if y_train_full.mean() > 0.5 else 0] * len(y_test))
    }

    knn = KNeighborsClassifier(n_neighbors=best_models_config["KNN"]["n_neighbors"])
    knn.fit(x_train_full, y_train_full)
    knn_pred = knn.predict_proba(x_test)[:, 1]
    knn_pred_class = knn.predict(x_test)
    knn_test_metrics = {
        "accuracy": accuracy_score(y_test, knn_pred_class),
        "precision": precision_score(y_test, knn_pred_class),
        "recall": recall_score(y_test, knn_pred_class),
        "f1_score": f1_score(y_test, knn_pred_class),
        "roc_auc_score": roc_auc_score(y_test, knn_pred)
    }

    lgr = LogisticRegression(C=best_models_config["LogisticRegression"]["C"])
    lgr.fit(x_train_full, y_train_full)
    lgr_pred = lgr.predict_proba(x_test)[:, 1]
    lgr_pred_class = lgr.predict(x_test)
    lgr_test_metrics = {
        "accuracy": accuracy_score(y_test, lgr_pred_class),
        "precision": precision_score(y_test, lgr_pred_class),
        "recall": recall_score(y_test, lgr_pred_class),
        "f1_score": f1_score(y_test, lgr_pred_class),
        "roc_auc_score": roc_auc_score(y_test, lgr_pred)
    }

    rf = RandomForestClassifier(n_estimators=30, max_depth=best_models_config["RandomForestClassifier"]["max_depth"])
    rf.fit(x_train_full, y_train_full)
    rf_pred = rf.predict_proba(x_test)[:, 1]
    rf_pred_class = rf.predict(x_test)
    rf_test_metrics = {
        "accuracy": accuracy_score(y_test, rf_pred_class),
        "precision": precision_score(y_test, rf_pred_class),
        "recall": recall_score(y_test, rf_pred_class),
        "f1_score": f1_score(y_test, rf_pred_class),
        "roc_auc_score": roc_auc_score(y_test, rf_pred)
    }

    nn = MLPClassifier(hidden_layer_sizes=best_models_config["MLPClassifier"]["hidden_layer_sizes"])
    nn.fit(x_train_full, y_train_full)
    nn_pred = nn.predict_proba(x_test)[:, 1]
    nn_pred_class = nn.predict(x_test)
    nn_test_metrics = {
        "accuracy": accuracy_score(y_test, nn_pred_class),
        "precision": precision_score(y_test, nn_pred_class),
        "recall": recall_score(y_test, nn_pred_class),
        "f1_score": f1_score(y_test, nn_pred_class),
        "roc_auc_score": roc_auc_score(y_test, nn_pred)
    }

    metrics_df = pd.DataFrame({
        "constant_answer": const_test_metrics,
        "KNN": knn_test_metrics,
        "LogisticRegression": lgr_test_metrics,
        "RandomForestClassifier": rf_test_metrics,
        "MLPClassifier": nn_test_metrics,
    })

    metrics_df.to_parquet(path_to_output_metrics)
