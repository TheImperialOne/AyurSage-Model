from flask import Flask, request, jsonify
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 

# Load ayurvedic_symptoms_desc.csv and formulation-Indications.csv
symptoms_data = pd.read_csv('ayurvedic_symptoms_desc.csv')
indications_data = pd.read_csv('Formulation-Indications.csv')

# Preprocess the data as needed
symptoms_data['Description'] = symptoms_data['Description'].str.lower()
indications_data['Main Indications'] = indications_data['Main Indications'].str.lower()

# Prepare the dataset for symptom prediction
X_symptom = symptoms_data['Description']
y_symptom = symptoms_data['Symptom']

# Vectorize descriptions using TF-IDF
tfidf_vectorizer_symptom = TfidfVectorizer()
X_symptom_tfidf = tfidf_vectorizer_symptom.fit_transform(X_symptom)

# Split the dataset into training and testing sets
X_train_symptom, X_test_symptom, y_train_symptom, y_test_symptom = train_test_split(
    X_symptom_tfidf, y_symptom, test_size=0.2, random_state=42
)

# Train a machine learning model to predict symptoms (e.g., MultinomialNB)
symptom_model = MultinomialNB()
symptom_model.fit(X_train_symptom, y_train_symptom)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the request
        user_input = request.json
        user_description = user_input['description'].lower()
        
        # Predict Symptom
        user_description_tfidf = tfidf_vectorizer_symptom.transform([user_description])
        predicted_symptom = symptom_model.predict(user_description_tfidf)[0]

        # Filter the data based on a partial match of the symptom in 'Main Indications'
        filtered_data = indications_data[indications_data['Main Indications'].str.lower().str.contains(predicted_symptom.lower())]

        # Get the list of predicted medicines
        predicted_medicines = list(filtered_data['Name of Medicine'])[:5]  # Limit to the top 5 medicines
        
        response = {
            "predicted_symptom": predicted_symptom,
            "predicted_medicines": predicted_medicines
        }

        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host= '0.0.0.0')
