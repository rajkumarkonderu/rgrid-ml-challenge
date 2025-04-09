from flask import Flask, jsonify, request
from typing import Literal
import pickle

app = Flask(__name__)

LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]

# Load model pipeline (includes TF-IDF + DecisionTreeClassifier)
with open("decision_tree_text_model.pkl", "rb") as f:
    model = pickle.load(f)

def predict(description: str) -> LABELS:
    """
    Predict the disease label given a trial description.
    """
    if not description.strip():
        raise ValueError("Description cannot be empty.")
    
    prediction = model.predict([description])[0]

    if prediction not in LABELS.__args__:
        raise ValueError("Prediction is not a valid label.")
    
    return prediction


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def identify_condition():
    data = request.get_json(force=True)

    try:
        prediction = predict(data["description"])
        return jsonify({"prediction": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run()
