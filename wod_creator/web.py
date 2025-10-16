from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests as req
from prompt_file import prompts

from use import run

app = Flask(__name__)
CORS(app)

print(prompts['wodprompt'])

OLLAMA_URL = "http://localhost:11434/api/generate"


@app.route("/")
def index():
   return send_from_directory("static","wod_oneri.html")

@app.route('/generate', methods=['POST'])
def generate():
   data = request.json
   #print(data) #this is a json with prompt
   
   
   try:
      name = data['myName']
      prompt = prompts[name]
      ans = run(prompt)
      #return jsonify(ans)
      return ans
   except Exception as e:
      return jsonify({"error": str(e)}), 500
   
if __name__ == "__main__":
   app.run(port=5000)




