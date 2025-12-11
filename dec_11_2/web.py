from flask import Flask, render_template, request
from ol import ask_ai
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html")

@app.route("/ai", methods=['POST'])
def ai():
    dream = request.form["dream"]
    f = open("prompts.txt","r")
    prompt = ""
    lines = f.readlines()
    for line in lines:
        prompt += line
    resp = ask_ai(prompt + " " + dream)

    mydict = json.loads(resp) #turns str into a dictionary item
    print(mydict)
    
    return render_template("dream.html", dream=dream, data=mydict["dream"], data2= mydict["full_analysis"])

if __name__ == "__main__":
    app.run(debug=True)