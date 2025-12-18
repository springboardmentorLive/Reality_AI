from groq import Groq

client = Groq(api_key="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")

def ask_bot(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

print("AI chatbot Ready! Type 'quit' to exit.")

while True:
    user = input("You: ")
    if user.lower() == 'quit':
        print("Bot: Goodbye!")
        break
    print("Bot:", ask_bot(user))
