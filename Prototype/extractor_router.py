#extractor_router.py
from extractors.aadhar_extractor import extract_aadhar_info
from extractors.pan_extractor import extract_pan_info
from extractors.resume_extractor import extract_resume_info
from extractors.passport_extractor import extract_passport_info
from extractors.license_extractor import extract_license_info
from extractors.invoice_extractor import extract_invoice_info
from extractors.passbook_extractor import extract_passbook_info

def route_extraction(label: str, doc_text: str) -> str:
    label_map = {
        "Aadhar Card": extract_aadhar_info,
        "PAN Card": extract_pan_info,
        "Resume": extract_resume_info,
        "Indian Passport (Republic of India)": extract_passport_info,
        "License": extract_license_info,
        "Invoice": extract_invoice_info,
        "Bank Passbook": extract_passbook_info
    }

    extractor = label_map.get(label)
    if extractor:
        return extractor(doc_text)
    else:
        return "No extractor available for this document type."
