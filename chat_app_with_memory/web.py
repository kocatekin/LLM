from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import requests as req
from figures import conversations



app = Flask(__name__)
CORS(app)

OLLAMA_URL = "http://localhost:11434/api/generate"
MAX_TURNS = 10


def run(prompt, model='llama3:8b'):
   resp = req.post(
      'http://localhost:11434/api/generate',
      json = {
         'model':model,
         'prompt':prompt,
         'stream': False
      }
   )
   try:
      ans = resp.json()['response']
      return ans
   except Exception as e:
      return e
   

def build_prompt(history):
   prompt = ""
   for msg in history:
      if msg["role"] == "system":
         prompt += f"System: {msg['content']}\n"
      elif msg["role"] == "user":
         prompt += f"User: {msg['content']}\n"
      else:
         prompt += f"Assistant: {msg['content']}\n"
   prompt += "Assistant: "
   return prompt




@app.route("/")
def index():
   return send_from_directory("static","index.html")

@app.route("/chat", methods=['POST'])
def chat():
   global conversation

   data = request.get_json()
   user_message = data.get("message", "")
   figure = data.get("figure", "Default")
   #print(user_message, figure)

   # if figure is not in our list
   if figure not in conversations:
      conversations[figure] = [
         {"role": "system", "content": f"You are {figure}, a historical figure. Stay in character and respond as {figure} would."}
      ]
   conversation = conversations[figure]

   #add user msg to history
   conversation.append({"role": "user", "content": user_message})
   print(conversation)

   #trim history to last N turns + system
   keep = conversation[:1] + conversation[-(MAX_TURNS*2):]

   #build prompt and get resp.
   prompt = build_prompt(keep)
   answer = run(prompt)

   # add ass. reply to history
   conversation.append({"role": "assistant", "content": answer})
   
   return jsonify({"response": answer})
   
if __name__ == "__main__":
   app.run(port=5000)

