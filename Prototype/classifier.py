#classifier.py
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
load_dotenv()

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are a document classification assistant. Your task is to classify the following document into **exactly one** of the following types:

- Resume
- Invoice
- Aadhar Card
- PAN Card
- License
- Indian Passport (Republic of India)
- Bank Passbook

You MUST respond with **only one of the labels** from the list above. Do NOT explain or provide any justification. If uncertain, pick the closest match.
Do NOT include explanations or any extra text.

Here are examples:

---

Document:
"Income Tax Department
Permanent Account Number: ATQPR1234L
Name: Renu Sharma"
Label: PAN Card

---

Document:
"Republic of India
Ministry of External Affairs
Passport No: R1234567
Name: Neha Singh"
Label: Indian Passport (Republic of India)

---

Document:
"State Bank of India
Account Number: 1234567890
Transaction:
- ₹500 credited
- ₹100 debited"
Label: Bank Passbook

---

Now classify the following document:

Document:
{doc_text}

Only respond with **one of the exact labels** from the list.
Your response must be just one of the above labels, nothing else.

""")


chain = prompt | llm

def classify_document(text: str) -> dict:
    response = chain.invoke({"doc_text": text})
    label = response.content.strip()
    
    # Return as a JSON-style Python dictionary
    return {
        "document_type": label
    }