from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are a document parser for Indian Driving Licenses. Given a noisy OCR text from a driving license, extract the following fields:

- Name
- Date of Birth (format: DD/MM/YYYY)
- License Number (e.g., MH12 20110012345 or similar format)
- Issue Date
- Expiry Date
- Father's Name
- Address (brief version)
- Blood Group (if found)

If any field is missing or unclear, use "Not Found".

Return the result as a compact single-line JSON object like:
{{"Name": "Rajesh Kumar", "Date of Birth": "01/01/1990", "License Number": "DL0420110012345", "Issue Date": "01/01/2011", "Expiry Date": "01/01/2031", "Father's Name": "Ramesh Kumar", "Address": "123 MG Road, Delhi", "Blood Group": "O+"}}

Text:
{doc_text}
""")

chain = prompt | llm

def extract_license_info(text: str) -> str:
    response = chain.invoke({"doc_text": text})
    return response.content.strip()
