import json
from pathlib import Path
from typing import List

from langchain.schema import Document


def documents_to_json(documents: List[Document], filename: str | Path) -> None:
    """
    Convert a list of Document objects to a JSON file.

    Args:
        documents (List[Document]): List of Document objects to be converted
        filename (str | Path): Path where the JSON file will be saved

    Raises:
        RuntimeError: If there's an error while writing the file
    """
    if isinstance(filename, str):
        filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    try:
        with filename.open("w", encoding="utf-8") as f:
            json.dump(
                [
                    {"page_content": doc.page_content, "metadata": doc.metadata}
                    for doc in documents
                ],
                f,
                indent=2,
            )
    except (PermissionError, IOError, OSError) as e:
        raise RuntimeError(f"Error while exporting JSON file: {e}")


def json_to_documents(filename: str | Path) -> List[Document]:
    """
    Convert a JSON file to a list of Document objects.

    Args:
        filename (str | Path): Path to the JSON file to be converted

    Returns:
        List[Document]: List of Document objects created from the JSON file

    Raises:
        RuntimeError: If there's an error while reading the file
        ValueError: If the JSON format is invalid
    """
    if isinstance(filename, str):
        filename = Path(filename)
    try:
        with filename.open("r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                documents = [
                    Document(
                        page_content=page["page_content"], metadata=page["metadata"]
                    )
                    for page in data
                ]
                return documents
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON format in {filename}: {e}")
    except (PermissionError, IOError, OSError) as e:
        raise RuntimeError(f"Error while importing JSON file: {e}")
