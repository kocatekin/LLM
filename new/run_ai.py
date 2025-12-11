# runai.py
import ollama

def ask_ai(prompt, model="llama3:8b"):
    """Simple non-streaming call to Ollama."""
    response = ollama.generate(
        model=model,
        prompt=prompt
    )
    return response["response"]

