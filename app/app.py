import os

from dotenv import load_dotenv
from flask import Flask, jsonify, request

from app.model.model import load_model

load_dotenv()
app = Flask(__name__)


@app.route("/")
def root():
    return "A web app to predict positive/negative sentiment"


@app.route("/predict")
def predict():
    model = load_model(os.getenv("PATH_MODEL"), compress=True)
    y_pred = model.predict_proba([request.args["text"]])[0]

    return jsonify(dict(zip(["negative", "positive"], y_pred.tolist())))
