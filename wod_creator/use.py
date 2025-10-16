import requests as req


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

if __name__ == "__main__":
   
   model = "gemma:2b"
   import random

   # Prompt to send -- no turkish
   prompt = f"as {random.choices(['donald trump','snoop dogg','kim kardashian', 'david goggins'])}, give honest advice to someone who is having a bad day. make it short, two sentences"
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
   print(f"🤖 {model} says:\n")
   print(response.json()['response'])
