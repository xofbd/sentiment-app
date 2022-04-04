import gzip
from html import unescape

import dill
import pandas as pd
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


def prepare_data(path_data):
    """Returning train/test splits of data set."""
    df = pd.read_csv(path_data, error_bad_lines=False)
    X = df["SentimentText"]
    y = df["Sentiment"]

    return train_test_split(X, y, test_size=0.2, random_state=0)


def construct_model(X_train, y_train, tokenizer=None):
    """Return trained ML model."""
    vectorizer = HashingVectorizer(
        preprocessor=lambda doc: unescape(doc).lower(),
        tokenizer=tokenizer,
        alternate_sign=False,
        stop_words="english"
    )
    clf = MultinomialNB()
    param_grid = {"alpha": range(1, 21)}
    grid_search = GridSearchCV(clf, param_grid, cv=5, verbose=1)

    model = Pipeline([("vectorizer", vectorizer), ("classifier", grid_search)])

    return model.fit(X_train, y_train)


def evaluate_model(model, name, X_train, X_test, y_train, y_test):
    """Print model performances"""
    print(f"Results for {name} model.")
    print(f"Training accuracy: {model.score(X_train, y_train) :g}")

    try:
        print(f"Cross-validation accuracy: {model[-1].best_score_ :g}")
    except AttributeError:
        pass
    else:
        for param, val in model[-1].best_params_.items():
            print(f"Best {param}: {val}")

    print(f"Testing accuracy: {model.score(X_test, y_test) :g}")


def serialize_model(model, path, compress=False):
    """Serialize scikit-learn model"""
    if compress:
        open_func = gzip.open
    else:
        open_func = open

    with open_func(path, "wb") as f:
        dill.dump(model, f, recurse=True)


def load_model(path, compress=False):
    """Load and return trained ML model"""
    if compress:
        open_func = gzip.open
    else:
        open_func = open

    with open_func(path, "rb") as f:
        return dill.load(f)


def main(path_data, path_model, compress=False):
    X_train, X_test, y_train, y_test = prepare_data(path_data)
    model = construct_model(X_train, y_train)
    evaluate_model(model, "Naive Bayes", X_train, X_test, y_train, y_test)
    serialize_model(model, path_model, compress=compress)


if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Create and serialize ML model.")
    parser.add_argument("path_data", type=str, help="path to training data")
    parser.add_argument("path_model", type=str, help="path to dump model")
    parser.add_argument(
        "-c",
        "--compress",
        action="store_true",
        default=False,
        help="whether to compress file using gzip"
    )
    args = parser.parse_args()

    main(args.path_data, args.path_model, compress=args.compress)
