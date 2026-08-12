# Q10. Mini Project – Multi-Model Assistant

from dotenv import load_dotenv
import os
from langchain_groq import ChatGroq
from langchain_mistralai import ChatMistralAI

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
mistral_key = os.getenv("MISTRAL_API_KEY")

while True:
    choice = input("\nChoose model (Groq/Mistral) or 'quit': ")

    if choice.lower() == "quit":
        print("Assistant closed.")
        break

    if choice.lower() == "groq":
        if not groq_key:
            print("Groq API key is missing.")
            continue
        model = ChatGroq(model="llama-3.1-8b-instant", max_tokens=150)

    elif choice.lower() == "mistral":
        if not mistral_key:
            print("Mistral API key is missing.")
            continue
        model = ChatMistralAI(model="mistral-small-2603", max_tokens=150)

    else:
        print("Invalid choice.")
        continue

    question = input("You: ")

    try:
        response = model.invoke(question)
        print("Assistant:", response.content)
    except Exception as e:
        print("Model error:", e)