from groq import Groq

client = Groq(api_key="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")

def ask_bot(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    # FIX: message ["content"] X message.content ✔
    return response.choices[0].message.content

print("AI Chatbot Ready! Type 'quit' to exit.\n")

while True:
    user = input("You: ")
    if user.lower() == "quit":
        print("Bot: Bye!")
        break
    print("Bot:", ask_bot(user))