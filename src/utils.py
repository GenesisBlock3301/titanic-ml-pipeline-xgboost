import pandas as pd

def load_data(path='../data/train.csv'):
    return pd.read_csv(path)


def clean_data(df):
    df = df.drop(['Cabin', 'Ticket', 'Name', 'PassengerId'], axis=1)
    return df

