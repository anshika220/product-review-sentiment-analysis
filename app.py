from flask import Flask, render_template, request, jsonify
import pickle
import re
from nltk.stem import PorterStemmer
import numpy as np

app = Flask(__name__)

# Load the model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vector.pkl", "rb"))

# Function to preprocess text and predict sentiment
def single_prediction(model, vectorizer, text_input):
    stemmer = PorterStemmer()

    # Preprocess the input text
    # 1. Convert to lowercase
    text = text_input.lower()

    # 2. Remove special characters and digits
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # 3. Tokenize the text using split method
    words = text.split()

    # 4. Apply stemming
    stemmed_words = [stemmer.stem(word) for word in words]

    # 5. Rejoin the stemmed words into a single string
    preprocessed_text = " ".join(stemmed_words)

    # Convert the preprocessed text to the required input format (numerical data)
    text_array = vectorizer.transform([preprocessed_text])  # Use the vectorizer to convert to numeric format

    # Make prediction using the model
    prediction = model.predict(text_array)

    # Interpret the model's output
    if prediction[0] == 'positive':
        return "The comment is positive."
    elif prediction[0] == 'negative':
        return "The comment is negative."
    else:
        return "The comment is neutral."


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user input from the form
        user_input = request.form.get("user_input")
        
        print(f"Received user input: {user_input}")  # Debugging line
        
        if not user_input:
            return jsonify({"error": "No input text provided!"}), 400
        
        # Call the single_prediction function
        prediction = single_prediction(model, vectorizer, user_input)
        
        print(f"Prediction result: {prediction}")  # Debugging line
        
        # Return prediction as JSON
        return jsonify({"prediction": prediction})
    except Exception as e:
        print(f"Error: {str(e)}")  # Debugging line
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)

