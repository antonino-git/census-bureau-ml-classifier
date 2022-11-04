# Script to train machine learning model.

from sklearn.model_selection import train_test_split
from ml.data import process_data
from ml.model import compute_model_metrics_by_slice, train_model, compute_model_metrics, inference, store_model, store_process_data_cfg

import pandas as pd
import csv

INPUT_DATASET_PATH = '../data/census_cleaned.csv'
MODEL_PATH = '../model/model.pkl'
ENCODER_PATH = '../model/encoder.pkl'
LB_PATH = '../model/lb.pkl'
MODEL_METRICS_REPORT_PATH = '../model/model_metrics.csv'

# Load the input dataset
data = pd.read_csv(INPUT_DATASET_PATH)

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
train, test = train_test_split(data, test_size=0.20)

cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

# Preprocess train data
X_train, y_train, encoder, lb = process_data(
    train, categorical_features=cat_features, label="salary", training=True)

# Train model
model = train_model(X_train, y_train)

# Save model
store_model(model, MODEL_PATH)

# Store the configuration parameters used for process the training data
store_process_data_cfg(encoder, ENCODER_PATH, lb, LB_PATH)

# Preprocess test data
X_test, y_test, _, _ = process_data(
    test, categorical_features=cat_features, label="salary", training=False,
    encoder=encoder, lb=lb)

# Inference with test data
y_pred = inference(model, X_test)

# Evaluate model performance
precision, recall, fbeta = compute_model_metrics(y_test, y_pred)
print(
    "Model performance:\nPrecision:{:.3f}\nRecall:{:.3f}\nfbeta:{:.3f}".format(
        precision,
        recall,
        fbeta))

# Evaluate model performance by slice
cat_metrics_list = compute_model_metrics_by_slice(
    model=model,
    X=test,
    cat_features=cat_features,
    label="salary",
    encoder=encoder,
    lb=lb)

with open(MODEL_METRICS_REPORT_PATH, "w") as model_report_f:
    model_report_writer = csv.DictWriter(
        model_report_f,
        fieldnames=[
            'category_feature',
            "category",
            "precision",
            "recall",
            "fbeta",
            "num_elements"])
    model_report_writer.writeheader()
    model_report_writer.writerows(cat_metrics_list)
