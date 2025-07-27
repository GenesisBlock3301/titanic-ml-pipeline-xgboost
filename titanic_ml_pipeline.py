from src.model import build_pipeline
from src.utils import load_data, clean_data
from sklearn.model_selection import cross_val_score


def main():
    path = './data/train.csv'
    df = load_data(path)
    df = clean_data(df)
    X = df.drop('Survived', axis=1)
    y = df.Survived

    pipeline = build_pipeline(X)
    scores = cross_val_score(pipeline, X, y, cv=5)

    print(f"Cross val score: {scores}")
    print(f"Cross val Avg score: {scores.mean()}")


if __name__ == "__main__":
    main()