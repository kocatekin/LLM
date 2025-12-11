from run_ai import ask_ai

def generate_todo_items():
    prompt = "I want you to generate todo items for me, these items should be about studying for an NLP course, limit to 5. However, make them fun. Do not put any preamble. Generate a proper json"
    res = ask_ai(prompt)
    print(res)


generate_todo_items()