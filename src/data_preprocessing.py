from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess_features(X):
    """
    Create preprocessing pipeline for numerical and categorical features.
    :param X: training features
    :return:
    """
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ['int64', 'float64']]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == 'object']

    # Just fine missing value and fill with median of col
    numeric_transform = SimpleImputer(strategy='median')

    # first impute categorical data with most frequent data, then apply one hot on X
    categorical_transform = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Apply above transformer on my dataset's relevant cols
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transform, numerical_cols),
            ('cat', categorical_transform, categorical_cols)
        ]
    )
    return preprocessor





