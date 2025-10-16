from flask import Flask, jsonify, request
from flask_cors import CORS
import requests as req

from use import run

app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"

@app.route('/generate', methods=['POST'])
def generate():
   data = request.json
   print(data) #this is a json with prompt
   if not data or "prompt" not in data:
      return jsonify({"error":"missing prompt"}), 400
   
   try:
      print(data['prompt'])
      ans = run(data['prompt'])
      print(ans)
      return jsonify(ans)
   except Exception as e:
      return jsonify({"error": str(e)}), 500
   
if __name__ == "__main__":
   app.run(port=5000)

