from flask import Flask, request, jsonify, render_template
# from flask_cors import CORS
import json
import random
import joblib

app = Flask(__name__)

model = joblib.load('intent_classifier.joblib')

with open('intents.json', 'r') as file:
    intents_data = json.load(file)['intents']

intent_tags = [intent['tag'] for intent in intents_data]
responses = {intent['tag']: intent['responses'] for intent in intents_data}


def get_response(user_input):
    try:
        predicted_intent = model.predict([user_input])[0]
    except Exception as e:
        print(f"Prediction error: {e}")
        return "Sorry, I'm having trouble understanding. Please try again."
    
    if predicted_intent not in intent_tags:
        return "Please specify your symptom (e.g., headache, sore throat)."
    
    try:
        return random.choice(responses[predicted_intent])
    except KeyError:
        return "I need to update my knowledge. Please ask something else."
    

@app.route('/chat', methods=['POST'])
def chat():
    user_message = request.json['message']
    bot_response = get_response(user_message)
    return jsonify({'reply': bot_response})

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)