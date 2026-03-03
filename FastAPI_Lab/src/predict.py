import joblib


def predict_wine_quality(X):
    model = joblib.load("../model/wine_model.pkl")
    y_pred = model.predict(X)
    return y_pred
