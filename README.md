#RGrid Machine Learning Challenge

This project is a text classification solution for the RGrid Machine Learning Challenge. It uses a machine learning pipeline to predict medical conditions from clinical trial descriptions and serves predictions through a Flask API.

## Project Structure
. ├── main.py # Flask API server ├── test.py # Test script for the API ├── model_training.py # Training script (optional) ├── decision_tree_text_model.joblib # Trained model file ├── trials.csv # Provided dataset ├── requirements.txt └── README.md


##Model

- **Model**: Decision Tree Classifier
- **Text Features**: TF-IDF Vectorizer
- **Pipeline**: `TfidfVectorizer` → `DecisionTreeClassifier`
- **Labels**:
  - Dementia
  - ALS
  - Obsessive Compulsive Disorder
  - Scoliosis
  - Parkinson’s Disease

## How to Run
Run the Flask API
python main.py
python test.py

### Install dependencies
pip install -r requirements.txt
flask
scikit-learn
pandas
numpy
joblib
requests



