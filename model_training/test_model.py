import pytest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from ml.data import process_data
from ml.model import train_model, inference, compute_model_metrics, store_model
from ml.model import load_model, store_process_data_cfg, load_process_data_cfg

CAT_FEATURES = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

TRAIN_DATA_PATH = "../data/census_cleaned.csv"
MODEL_PATH = '../model/model.pkl'
ENCODER_PATH = '../model/encoder.pkl'
LB_PATH = '../model/lb.pkl'


@pytest.fixture(scope="module")
def data():

    train_dataset = pd.read_csv(TRAIN_DATA_PATH)
    train, test = train_test_split(train_dataset, test_size=0.20)

    # Preprocess train data
    X_train, y_train, encoder, lb = process_data(
        train, categorical_features=CAT_FEATURES, label="salary",
        training=True)

    # Preprocess test data
    X_test, y_test, _, _ = process_data(
        test, categorical_features=CAT_FEATURES, label="salary",
        training=False, encoder=encoder, lb=lb)

    data = {"X_train": X_train, "y_train": y_train, "X_test": X_test,
            "y_test": y_test, "encoder": encoder, "lb": lb}

    return data


@pytest.fixture(scope="module")
def model(data):
    # Train model
    model = train_model(data['X_train'], data['y_train'])

    return model


def test_inference_len(data, model):
    """ Check that the inference output has the right size """

    y_pred = inference(model, data['X_test'])

    assert len(y_pred) == len(data['X_test'])


def test_compute_model_metrics_type(data, model):
    """ Check that the compute_model_metrics outputs are the right type """

    y_pred = inference(model, data['X_test'])

    m1, m2, m3 = compute_model_metrics(data['y_test'], y_pred)

    assert isinstance(
        m1,
        float) and isinstance(
        m2,
        float) and isinstance(
            m3,
        float)


def test_load_store_model(data, model):
    """ Check that the loaded_model reads the right model """

    store_model(model, MODEL_PATH)

    loaded_model = load_model(MODEL_PATH)

    y_pred_original = inference(model, data['X_test'])
    y_pred_loaded = inference(loaded_model, data['X_test'])

    assert np.array_equal(y_pred_original, y_pred_loaded)


def test_load_store_process_cfg(data):
    """ Check that the store_process_data_cfg and load_process_data_cfg work
        as expected """

    store_process_data_cfg(data['encoder'], ENCODER_PATH, data['lb'], LB_PATH)

    encoder, lb = load_process_data_cfg(ENCODER_PATH, LB_PATH)

    assert isinstance(
        encoder, type(
            data['encoder'])) and (
        isinstance(
            lb, type(
                data['lb'])))
