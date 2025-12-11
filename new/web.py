# web.py
from flask import Flask, request, jsonify, Response, render_template
from run_ai import ask_ai

app = Flask(__name__)

# Render HTML
@app.route("/")
def home():
    return render_template("index.html")

# Regular AI call
@app.route("/api/ask", methods=["POST"])
def api_ask():
    data = request.get_json()
    prompt = data.get("prompt", "")
    txt = ask_ai(prompt)
    print(txt)
    return jsonify({"response": txt})



if __name__ == "__main__":
    app.run(debug=True)
