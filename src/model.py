from sklearn.pipeline import Pipeline

from src.data_preprocessing import preprocess_features
from xgboost import XGBRegressor


def build_pipeline(X):
    preprocessor = preprocess_features(X)

    model = XGBRegressor(n_estimators=100, learning_rate=0.1, eval_metric='rmse')

    clf = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return clf
