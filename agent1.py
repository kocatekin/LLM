import requests as req

def run(prompt, model='gemma:2b'):
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
      mydict = {}
      mydict['ans'] = ans
      return mydict
   except Exception as e:
      return e

if __name__ == "__main__":
   
   model = "gemma:2b"
   import random

   # Prompt to send -- no turkish
   prompt = "you are a code generator. return only valid python code. do not explain, do not add ```. output must be executable by python interpreter. write a python program to print hello world"
   print(f"{prompt}")

   # Send POST request to local Ollama server
   response = req.post(
      'http://localhost:11434/api/generate',
      json={
         'model': model,
         'prompt': prompt,
         'stream': False
      }
   )

   # Print the model's reply
   print(f"🤖 {model}:\n")
   #print(response.json()['response'])

   resp = response.json()['response']
   #print(resp)
   if(resp.startswith("```")):
      resp = resp.split("```",1)[1]
      if "```" in resp:
         resp = resp.split("```",1)[0]
   if(resp.startswith("python")):
      resp = resp.split("python")[1]
   print(resp)
   exec(resp.strip())




