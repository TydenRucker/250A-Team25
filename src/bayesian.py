import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from read_data import load_data

def preprocess(chord_df, chroma_df):
    X = []
    y = []

    songs = chroma_df.index.levels[0]
    for song in songs:
        chroma_data = chroma_df.loc[song].copy()
        chord_data = chord_df.loc[song].copy()
        chroma_data = chroma_data.sort_values('time')
        chord_data = chord_data.sort_values('start')

        merged = pd.merge_asof(
            chroma_data,
            chord_data,
            left_on = "time",
            right_on = "start",
            direction = "backward"
        )

        X.append(merged[list(range(24))])
        y.append(merged["chord_id"])

    X = np.vstack(X)
    y = np.concatenate(y)

    return X, y

if __name__ == "__main__":
    chord_df, chroma_df = load_data()
    X, y = preprocess(chord_df, chroma_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    model = GaussianNB()

    print("training bayesian")
    model.fit(X_train, y_train)

    print("predicting")
    y_pred = model.predict(X_test)

    print("accuracy:", accuracy_score(y_test, y_pred))
