# web.py
from flask import Flask, request, jsonify, Response, render_template
from run_ai import ask_ai
from prompts import system_prompt

app = Flask(__name__)

SYSPROMPT = system_prompt



# Render HTML
@app.route("/")
def home():
    return render_template("index.html")

# Regular AI call
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    print(data)
    user_prompt = data.get("user_prompt", "")
    prompt = generate_prompt(SYSPROMPT, user_prompt)
    txt = ask_ai(prompt)
    return jsonify({"response": txt})

# generate prompt here

def generate_prompt(initial, message):
    return f"{SYSPROMPT}, {message}"

if __name__ == "__main__":
    app.run(debug=True)
