import ollama

def ask_ai(prompt, model="llama3.2:latest"):
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    return response["response"]