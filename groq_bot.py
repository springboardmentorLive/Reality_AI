import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class ChatBot:
    def __init__(self, api_key, model="llama-3.3-70b-versatile", max_history=6):
        self.client = Groq(api_key=api_key)
        self.model = model
        self.max_history = max_history

        self.system_prompt = (
            "You are a professional, concise, and technically accurate AI assistant. "
            "Respond clearly. Avoid unnecessary verbosity."
        )

        self.history = [
            {"role": "system", "content": self.system_prompt}
        ]

    def ask(self, user_input):
        # Add user message
        self.history.append({"role": "user", "content": user_input})

        # Trim history to avoid context explosion
        self._trim_history()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.history,
                temperature=0.7
            )

            reply = response.choices[0].message.content

            # Add assistant response to memory
            self.history.append({"role": "assistant", "content": reply})

            return reply

        except Exception as e:
            return f"[Error] {str(e)}"

    def _trim_history(self):
        """
        Keep system message + last N interactions
        """
        if len(self.history) > 1 + self.max_history * 2:
            self.history = (
                [self.history[0]] +
                self.history[-self.max_history * 2:]
            )


# ------------------------
# Usage
# ------------------------
if __name__ == "__main__":
    api_key = os.environ.get("PUBLIC_GROQ_API_KEY")
    if not api_key:
        raise EnvironmentError("PUBLIC_GROQ_API_KEY not set")

    bot = ChatBot(api_key)

    print("AI Chatbot Ready. Type 'quit' to exit.\n")

    while True:
        user = input("You: ")
        if user.lower() == "quit":
            print("Bot: Goodbye.")
            break

        response = bot.ask(user)
        print("Bot:", response)
