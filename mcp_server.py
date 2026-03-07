from mcp.server.fastmcp.prompts import base
from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP("DocumentMCP", log_level="ERROR")


docs = {
    "deposition.md": "This deposition covers the testimony of Angela Smith, P.E.",
    "report.pdf": "The report details the state of a 20m condenser tower.",
    "financials.docx": "These financials outline the project's budget and expenditures.",
    "outlook.pdf": "This document presents the projected future performance of the system.",
    "plan.md": "The plan outlines the steps for the project's implementation.",
    "spec.txt": "These specifications define the technical requirements for the equipment.",
}


@mcp.tool(
    name="read_doc_contents",
    description="Reads the contents of a document and return it as a string",
)
def read_doc_contents(doc_id: str = Field(description="The ID of the document to read")):
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    return docs[doc_id]


@mcp.tool(
    name="edit_document",
    description="Edit a document by replacing a string in the documents content with a new string.",
)
def edit_document(
    doc_id: str = Field(description="The ID of the document to edit"),
    old_string: str = Field(
        description="The string to be replaced in the document, must match exactly including whitespace"
    ),
    new_string: str = Field(description="The new text to replace the old text"),
):
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    docs[doc_id] = docs[doc_id].replace(old_string, new_string)


@mcp.resource(
    "docs://documents",
    mime_type="application/json",
    name="list_docs",
    description="Returns a list of all document IDs available in the system.",
)
def list_docs() -> list[str]:
    return list(docs.keys())


@mcp.resource(
    "docs://documents/{doc_id}",
    mime_type="text/plain",
    name="fetch_doc",
    description="Returns the content of a document given its ID.",
)
def fetch_doc(doc_id: str) -> str:
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    return docs[doc_id]


@mcp.prompt(
    name="format",
    description="Rewrites a document in markdown format. Input should be the ID of the document to rewrite.",
)
def format_doc(doc_id: str) -> list[base.Message]:
    if doc_id not in docs:
        raise ValueError(f"Document with ID '{doc_id}' not found.")
    prompt = f"""
        Your goal is to reformat a document to be written with markdown syntax.
        The id of the document you need to reformat is:
        <document_id>
        {doc_id}
        </document_id>

        Here is the current content of the document:
        <document_content>
        {docs[doc_id]}
        </document_content>

        Add in headers, bullet points, tables, etc as necessary. Feel free to add in structure.
        Use the 'edit_document' tool to edit the document. After editing, return the full reformatted content of the document so the user can see it.
        """
    return [base.UserMessage(prompt)]


if __name__ == "__main__":
    mcp.run(transport="stdio")
