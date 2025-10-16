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
   
   import random
   #model = "gemma:2b"
   models = ["gemma:2b","llama3:8b"]
   for idx, model in enumerate(models):
      print(f"{idx}. {model}")
   choice = int(input("choose your model: "))
   model = models[choice]
   print(f"you chose: {model}")

   
   while True:
      
      prompt = input("> ")
      
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
