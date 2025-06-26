from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are a resume parser AI. Extract the following information from the OCR or plain text of a resume.

Extract the following fields:

- Name
- Email
- Phone Number
- Education (highest qualification with institution name)
- Skills (comma-separated list)
- Years of Experience (approximate number)
- Current Job Title (if available)

If any value is missing, return "Not Found".

Respond ONLY with a valid single-line JSON object like:
{{"Name": "Ravi Kumar", "Email": "ravi.kumar@example.com", "Phone Number": "+91-9876543210", "Education": "M.Tech in Computer Science from IIT Bombay", "Skills": "Python, Machine Learning, SQL", "Years of Experience": "3", "Current Job Title": "Data Scientist"}}

Text:
{doc_text}
""")

chain = prompt | llm

def extract_resume_info(text: str) -> str:
    response = chain.invoke({"doc_text": text})
    return response.content.strip()
