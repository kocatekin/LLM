import ollama

def run_local_model(system_prompt, user_prompt, model="llama3.2:latest"):
    response = ollama.chat(
        model=model,
        messages = [
            {"role":"system", "content":system_prompt},
            {"role":"user", "content": user_prompt}
        ]
    )
    return response["message"]["content"]


if __name__ == "__main__":
    system_prompt = input("enter system prompt \n")
    user_prompt = input("enter user prompt: \n")
    ans = run(system_prompt, user_prompt, model)
    print(ans)
