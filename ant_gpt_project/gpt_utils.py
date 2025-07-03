import openai
from dotenv import load_dotenv
import os

# Load .env file to get the OpenAI API key
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')

def ask_gpt_for_ant_decision(state_data):
    """
    Use structured prompt to decide if the ant moves toward food or randomly.
    state_data is a string: e.g. "Ant sees food nearby: True"
    """
    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use GPT-4 model
        messages=[
            {"role": "system", "content": "You decide how an ant should move in a foraging scenario."},
            {"role": "user", "content": f"{state_data} Should it move toward food or randomly? Reply with 'toward' or 'random'."}
        ]
    )
    return response.choices[0].message['content'].strip().lower()
