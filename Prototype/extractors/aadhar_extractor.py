#aadhar_extractor.py
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are an expert at extracting fields from noisy OCR output of Indian Aadhaar Cards.

From the text below, extract:

- Name (capitalize properly)
- Aadhaar Number (12-digit number, no spaces)
- Enrolment Number (XXXX/XXXXX/XXXXX format)
- Date of Birth or Year of Birth (DD/MM/YYYY or YYYY)
- Gender (Male/Female/Other)

If a value is missing, write "Not Found". 
Do not include line breaks or incomplete output.
Respond ONLY with a **valid compact JSON object** on a single line.

Example:
{{"Name": "John Doe", "Aadhaar Number": "123456789012", "Enrolment Number": "1234/12345/12345", "Date of Birth or Year of Birth": "1990", "Gender": "Male"}}

Text:
{doc_text}
""")

chain = prompt | llm

def extract_aadhar_info(text: str) -> str:
    response = chain.invoke({"doc_text": text})
    return response.content.strip()
