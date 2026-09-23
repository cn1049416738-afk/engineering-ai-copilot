import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

user_input = input("Ask me an engineering question: ")

response = client.responses.create(
    model="gpt-5.6-luna",
    instructions="""
You are an engineering copilot specialized in:
- mechanical engineering
- mechatronics
- control systems
- multiphysics simulation
- digital twins
- experimental data analysis

When answering:
1. Explain the engineering principle clearly.
2. If equations are involved, explain every variable.
3. For simulation problems, provide practical troubleshooting steps.
4. Distinguish assumptions from facts.
5. If uncertain, say so clearly.
6. Prefer concise but technically useful answers.
""",
    input=user_input
)


print("\nAI answer:")
print(response.output_text)
