from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

from read_data import load_data, preprocess

if __name__ == "__main__":
    print("loading data")
    chord_df, chroma_df = load_data()

    print("processing data for nb")
    X, y, _ = preprocess(chord_df, chroma_df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

    model = GaussianNB()

    print("training bayesian")
    model.fit(X_train, y_train)

    print("predicting")
    y_pred = model.predict(X_test)

    print("accuracy:", accuracy_score(y_test, y_pred))
