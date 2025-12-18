from groq import Groq

client = Groq(api_key="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")

def ask_bot(theme, style):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content":
                f"You are a poetic AI. Always reply in poetry form. The style must be {style}. "
                "Make your poem creative, beautiful, and emotionally expressive."
            },
            {
                "role": "user",
                "content": f"Write a poem about: {theme}"
            }
        ]
    )
    return response.choices[0].message.content

print("Poetry Generator Ready! Type 'quit' to exit.\n")

while True:
    theme = input("Enter a poem theme (love, nature, life, dreams, anything): ")

    if theme.lower() == "quit":
        print("Bot: Goodbye!")
        break

    style = input("Choose a style (romantic, sad, motivational, funny, dark, classical): ")

    print("\n--- Your Poem ---\n")
    print(ask_bot(theme, style))
    print("\n------------------\n")
