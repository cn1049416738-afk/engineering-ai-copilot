import os
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

pdf_path = "data/JIMSS2025_Modeling_of_abrasive_magnetorheological_fluids_for_drag_finishing_revision2.pdf"

reader = PdfReader(pdf_path)

paper_text = ""

for page in reader.pages[:3]:
    text = page.extract_text()
    if text:
        paper_text += text + "\n"

user_input = input("Ask me an engineering question: ")

prompt = f"""
Answer the question only using the paper content below.

Paper content:
{paper_text}

Question:
{user_input}

If the paper content does not contain enough information, say:
"The provided paper excerpt does not contain enough information."
"""

response = client.responses.create(
    model="gpt-5.6-luna",
    input=prompt
)

print("\nAI answer:")
print(response.output_text)