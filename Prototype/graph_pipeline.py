# graph_pipeline.py
from langgraph.graph import StateGraph, END
from typing import TypedDict
from classifier import classify_document
from extractor_router import route_extraction
from langsmith import traceable  

class DocState(TypedDict):
    doc: str
    label: str
    extracted_info: str

@traceable(name="Load Document")
def load_node(state: DocState) -> DocState:
    return {"doc": state["doc"]}

@traceable(name="Classify Document")
def classify_node(state: DocState) -> DocState:
    result = classify_document(state["doc"])
    return {"doc": state["doc"], "label": result["document_type"]}

@traceable(name="Extract Information")
def extract_node(state: DocState) -> DocState:
    extracted = route_extraction(state["label"], state["doc"])
    return {
        "doc": state["doc"],
        "label": state["label"],
        "extracted_info": extracted
    }

def build_graph():
    builder = StateGraph(state_schema=DocState)

    builder.add_node("load", load_node)
    builder.add_node("classify", classify_node)
    builder.add_node("extract", extract_node)

    builder.set_entry_point("load")
    builder.add_edge("load", "classify")
    builder.add_edge("classify", "extract")
    builder.add_edge("extract", END)

    return builder.compile()
