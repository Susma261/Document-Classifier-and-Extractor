from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are an expert at reading Indian passports from noisy OCR outputs.

Extract the following fields from the given passport text:

- Name (Full name as printed)
- Passport Number
- Date of Birth (in DD/MM/YYYY or YYYY-MM-DD)
- Gender
- Place of Birth
- Date of Issue
- Date of Expiry
- Father's Name
- Mother's Name
- Nationality
- File Number

If any field is missing or not found, return "Not Found".

Respond ONLY with a valid single-line compact JSON like this:
{{
  "Name": "Amit Sharma",
  "Passport Number": "M1234567",
  "Date of Birth": "1990-01-01",
  "Gender": "Male",
  "Place of Birth": "Delhi",
  "Date of Issue": "2015-01-01",
  "Date of Expiry": "2025-01-01",
  "Father's Name": "Rajesh Sharma",
  "Mother's Name": "Sunita Sharma",
  "Nationality": "Indian",
  "File Number": "DL1234567890123"
}}

Text:
{doc_text}
""")

chain = prompt | llm

def extract_passport_info(text: str) -> str:
    response = chain.invoke({"doc_text": text})
    return response.content.strip()
