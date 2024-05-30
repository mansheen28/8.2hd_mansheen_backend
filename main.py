from flask import Flask, request, jsonify
from datetime import datetime
import socket
import json
import argparse
import os
from gradientai import Gradient

app = Flask(__name__)
base_model = None
token = 'gFUwiECcl52VEZOUZE6RYALLu4QjwX2P'
workspace_id = '93af8759-ef81-4b11-8ed7-363ff2efc1cb_workspace'

os.environ['GRADIENT_ACCESS_TOKEN'] = token
os.environ['GRADIENT_WORKSPACE_ID'] = workspace_id

def prepareLlamaBot():
    global base_model
    gradient = Gradient()
    base_model = gradient.get_base_model(base_model_slug="llama3-8b-chat")

@app.route('/')
def index():
    return "Welcome to the Flask API!"

@app.route('/chat', methods=['POST'])
def chat():
    global base_model, user_message, chat_history
    data = request.get_json()

    if 'userMessage' not in data or not isinstance(data['userMessage'], str):
        return jsonify({'error': 'userMessage must be a string'}), 400
    
    if 'chatHistory' not in data or not isinstance(data['chatHistory'], list):
        return jsonify({'error': 'chatHistory must be a list'}), 400

    if not all(isinstance(item, dict) and 'User' in item and 'Llama' in item for item in data['chatHistory']):
        return jsonify({'error': 'chatHistory must be a list of dictionaries with keys User and Llama'}), 400

    user_input = data['userMessage']
    chat_history = data['chatHistory']
    chat_history_str = '\n'.join([f"{item['User']} - {item['Llama']}" for item in chat_history])

    QUERY = f"""
    [INST]
    YOU ARE A VIRTUAL ASSISTANT HELPING A USER NAVIGATE THEIR CURRENT SITUATION AND PROVIDING CONTEXTUALLY RELEVANT RECOMMENDATIONS.
    USE THE FOLLOWING GUIDELINES FOR THE INTERACTION:
    1. CONSIDER THE USER'S CIRCUMSTANCES AND LATEST QUERY.
    2. HOLD A CONTEXT-AWARE CONVERSATION.
    3. ADJUST THE COMPLEXITY OF YOUR RESPONSES BASED ON THE USER'S NEEDS AND THE SITUATION DESCRIBED.
    4. PROVIDE CONSTRUCTIVE AND HELPFUL ADVICE TAILORED TO THE SPECIFIC SITUATION.
    5. OFFER ENCOURAGEMENT AND SUGGEST PRACTICAL SOLUTIONS OR RECOMMENDATIONS.
    6. FOCUS ON PROVIDING REAL-TIME, CONTEXTUAL ASSISTANCE.
     
    GIVEN THE CHAT HISTORY:
    {chat_history_str}
    
    GIVEN THE USER'S INPUT:
    {user_input}

    RESPOND TO THE USER, CONTINUE THE CONVERSATION APPROPRIATELY, ADJUST THE COMPLEXITY AS PER THE USER'S NEEDS, AND PROVIDE RELEVANT AND USEFUL RECOMMENDATIONS. KEEP YOUR RESPONSE SHORT AND DIRECT.
    [/INST]
    """

    response = base_model.complete(query=QUERY, max_generated_token_count=500).generated_output

    return jsonify({'message': response}), 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=5000, help='Specify the port number')
    args = parser.parse_args()

    port_num = args.port
    prepareLlamaBot()
    print(f"App running on port {port_num}")
    app.run(port=port_num)
