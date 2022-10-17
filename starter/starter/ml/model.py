from sklearn.metrics import fbeta_score, precision_score, recall_score

from sklearn.ensemble import RandomForestClassifier

from ml.data import process_data
import pickle

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    rfc = RandomForestClassifier(n_jobs=-1)

    return rfc.fit(X_train, y_train)



def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def compute_model_metrics_by_slice(model, X, cat_features, label, encoder, lb):
    """
    Validates the trained machine learning model using precision, recall, and F1
    on test dataset slices.

    Inputs
    ------
    model:
        Trained model
    X : pd.DataFrame
        Dataframe containing the test dataset
    cat_features: list[str]
        List containing the names of the categorical features
    label : str
        Name of the label column in `X`.
    encoder : sklearn.preprocessing._encoders.OneHotEncoder
        Trained sklearn OneHotEncoder
    lb : sklearn.preprocessing._label.LabelBinarizer
        Trained sklearn LabelBinarizer

    Returns
    -------
    cat_metrics_list list[Dict]
    """

    cat_metrics_list = []

    for cat_feature in cat_features:
        for category in X[cat_feature].unique():


            X_cat = X[X[cat_feature] == category]

            x, y, _, _ = process_data(X_cat, categorical_features=cat_features,
                label=label, training=False, encoder=encoder, lb=lb)

            y_preds = inference(model, x)
            precision, recall, fbeta = compute_model_metrics(y, y_preds)

            cat_metrics = {}
            cat_metrics['category_feature'] = cat_feature
            cat_metrics['category'] = category
            cat_metrics['precision'] = precision
            cat_metrics['recall'] = recall
            cat_metrics['fbeta'] = fbeta
            cat_metrics['num_elements'] = len(X_cat)

            cat_metrics_list.append(cat_metrics)


    return cat_metrics_list


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    return model.predict(X)


def store_model(model, path):
    """ Store the machine learning model in path.

    Inputs
    ------
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    path: string
        The location where to store the model.
    Returns
    -------
    None
    """
    pickle.dump(model, open(path, 'wb'))

def load_model(path):
    """ Load the machine learning model from path

    Inputs
    ------
    path: string
        The location from where to load the model.
    Returns
    -------
    model : sklearn.ensemble._forest.RandomForestClassifier
        Trained machine learning model.
    """

    return pickle.load(open(path, 'rb'))