# models/deepseek_model.py
import os
from openai import OpenAI

def generate_reply(messages):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPENROUTER_API_KEY"),
    )
    extra_headers = {
        "HTTP-Referer": os.environ.get("SITE_URL", "http://localhost"),
        "X-Title": os.environ.get("SITE_NAME", "MySite")
    }
    try:
        completion = client.chat.completions.create(
            extra_headers=extra_headers,
            model="deepseek/deepseek-r1:free",
            messages=messages
        )
    except Exception as e:
        # Log the exception and return a fallback message
        print("Error calling Deepseek API:", e)
        return "I'm sorry, I'm having trouble generating a response right now."
    
    if not completion or not hasattr(completion, "choices") or not completion.choices:
        print("Invalid response from Deepseek:", completion)
        return "I'm sorry, I'm having trouble generating a response right now."
    
    first_choice = completion.choices[0]
    if not hasattr(first_choice, "message") or not first_choice.message:
        print("Invalid choice structure in Deepseek response:", first_choice)
        return "I'm sorry, I'm having trouble generating a response right now."
    
    return first_choice.message.content
