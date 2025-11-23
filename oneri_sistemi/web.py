from use import run_local_model
from flask import Flask, jsonify, request, send_from_directory

app = Flask(__name__)

SYSTEM_PROMPT = """
"""
USER_PROMPT = """"""

@app.route("/")
def index():
    return send_from_directory("static","index.html")


@app.route("/llm", methods=['POST'])
def generate():
    data = request.json
    print(data)
    if not data:
        return jsonify({"error":"no json body recevied"}), 400

    # tones here
    system_prompt = f'You are an excuse generator. You are a very {data["toSend"]["tone"]} person. Create a {data["toSend"]["tone"]} response. Just give the response. Make no explanations.'
    user_prompt = data["toSend"]["topic"]
    
    try:
        print("sending to local llm model")
        ans = run_local_model(system_prompt, user_prompt)
        print(ans)
        return jsonify(ans)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)
