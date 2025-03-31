import json
from pathlib import Path

from langchain.schema import Document


class DocumentJsonToolkit:
    """Toolbox class for converting Document objects to and from JSON files."""

    @staticmethod
    def documents_to_json(documents: list[Document], filename: str | Path) -> None:
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
        except (PermissionError, IOError) as e:
            raise RuntimeError(f"Error while exporting JSON file: {e}")

    @staticmethod
    def json_to_documents(filename: str | Path) -> list[Document]:
        if isinstance(filename, str):
            filename = Path(filename)
        try:
            with filename.open("r", encoding="utf-8") as f:
                data = json.load(f)
                documents = [
                    Document(
                        page_content=page["page_content"], metadata=page["metadata"]
                    )
                    for page in data
                ]
            return documents
        except (PermissionError, IOError) as e:
            raise RuntimeError(f"Error while importing JSON file: {e}")
