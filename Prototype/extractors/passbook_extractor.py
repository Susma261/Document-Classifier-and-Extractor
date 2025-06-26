from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are an expert at extracting fields from noisy OCR text of Indian Bank Passbooks.

From the given text, extract the following fields only:

- Account Holder Name
- Account Number
- IFSC Code (an 11-character alphanumeric code, e.g., SBIN0001234)
- Bank Name
- Branch Name
- Customer ID (if present)
- Opening Balance (if mentioned)

If anything is missing, return "Not Found".

Return the output as a compact single-line JSON. Example:

{{
  "Account Holder Name": "Ravi Kumar",
  "Account Number": "123456789012",
  "IFSC Code": "SBIN0001234",
  "Bank Name": "State Bank of India",
  "Branch Name": "MG Road Branch",
  "Customer ID": "9876543210",
  "Opening Balance": "₹5000.00"
}}

Text:
{doc_text}
""")

chain = prompt | llm

def extract_passbook_info(text: str) -> str:
    response = chain.invoke({"doc_text": text})
    return response.content.strip()
