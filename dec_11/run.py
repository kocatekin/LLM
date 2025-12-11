from ai import ask_ai
from flask import Flask, render_template
import json

app = Flask(__name__)


@app.route('/')
def index():
    return render_template("index.html", data="Hello!")

@app.route('/ai')
def ai():
    prompt = readfile()
    res = ask_ai(prompt)
    mydict = json.loads(res)
    return render_template("index.html", data=mydict)


#helpers 
def readfile():
    f = open("prompts.txt","r")
    lines = f.readlines()
    prompt = ""
    for line in lines:
        prompt += line
    return prompt

if __name__ == "__main__":
    app.run(debug=True)