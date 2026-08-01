import pandas as pd


def load_data():
    df = pd.read_csv("data/california_housing.csv")
    return df


if __name__ == "__main__":
    df = load_data()

    print("Dataset loaded successfully!")
    print(df.head())