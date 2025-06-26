from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI

llm = ChatMistralAI(temperature=0.1)

prompt = ChatPromptTemplate.from_template("""
You are an expert in reading international invoices from noisy OCR text.

From the text below, extract the following fields and return them as a valid **single-line JSON**:

- Invoice Number
- Invoice Date (format: DD/MM/YYYY or YYYY-MM-DD)
- Vendor Name
- Total Amount (include the currency symbol, e.g., ₹, $, €, £, etc.)
- GST Number or equivalent (if found)
- Items (each with Description, Quantity, Unit Price, Total Price — include currency symbol if available)

If a field is missing, return "Not Found".

Respond ONLY with a valid compact JSON like this:
{{"Invoice Number": "INV1234", "Invoice Date": "2023-05-12", "Vendor Name": "ABC Traders", "Total Amount": "$6543.21", "GST Number": "27ABCDE1234F1Z5", "Items": [{{"Description": "Product A", "Quantity": "2", "Unit Price": "$500", "Total Price": "$1000"}}]}}

Text:
{doc_text}
""")

chain = prompt | llm

def extract_invoice_info(text: str) -> str:
    response = chain.invoke({"doc_text": text})
    return response.content.strip()
