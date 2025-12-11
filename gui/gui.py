import tkinter as tk
from tkinter import ttk
from run_ai import ask_ai

def generate_text():
    f = open("prompt.txt","r")
    lines = f.readlines()
    prompt = ""
    for line in lines:
        prompt += line
        
    res = ask_ai(prompt)

    output_box.config(state="normal")
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, res)
    output_box.config(state="disabled")

def copy_text():
    text_to_copy = output_box.get("1.0", tk.END).strip()
    root.clipboard_clear()
    root.clipboard_append(text_to_copy)

root = tk.Tk()
root.title("Daily NPC Generator")
root.geometry("420x300")
root.resizable(False, False)

# Main Frame
main_frame = ttk.Frame(root, padding=20)
main_frame.pack(fill="both", expand=True)

# Title Label
title_label = ttk.Label(
    main_frame,
    text="Daily NPC Encounter Generator",
    font=("Segoe UI", 14, "bold")
)
title_label.pack(pady=(0, 15))

# Generate Button
generate_btn = ttk.Button(
    main_frame,
    text="Generate NPC",
    command=generate_text
)
generate_btn.pack(pady=(0, 10))

# Output Text Box (read-only)
output_box = tk.Text(
    main_frame,
    height=5,
    width=50,
    wrap="word",
    font=("Segoe UI", 11),
    state="disabled",
    relief="solid",
    borderwidth=1
)
output_box.pack(pady=(0, 10))

# Copy Button
copy_btn = ttk.Button(
    main_frame,
    text="Copy to Clipboard",
    command=copy_text
)
copy_btn.pack()

root.mainloop()
