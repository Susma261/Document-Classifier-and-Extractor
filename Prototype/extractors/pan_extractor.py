from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are an expert in parsing Indian PAN Cards from noisy OCR outputs. 

From the following text, extract the following fields as a JSON object:

- Name
- Father's Name
- Date of Birth (in DD/MM/YYYY format)
- PAN Number (format: ABCDE1234F)

If any field is missing, write "Not Found". Respond ONLY with a valid compact JSON object on a single line.

Example:
{{"Name": "Renu Sharma", "Father's Name": "Rajesh Sharma", "Date of Birth": "14/07/1985", "PAN Number": "ATQPR1234L"}}

Text:
{doc_text}
""")

chain = prompt | llm

def extract_pan_info(text: str) -> str:
    response = chain.invoke({"doc_text": text})
    return response.content.strip()
