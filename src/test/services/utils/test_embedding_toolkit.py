import pytest
from langchain.schema import Document

from src.services.utils.embedding_toolkit import (
    MAX_NAME_LENGTH,
    TRUNCATED_NAME_LENGTH,
    UUID_SUFFIX_LENGTH,
    create_chunk_ids,
    create_image_id,
    generate_safe_name,
    make_safe_slug,
    text_splitter_recursive_char,
)


@pytest.fixture
def sample_document():
    return Document(
        page_content="This is a test document",
        metadata={"source": "/path/to/sample-document.txt"},
    )


@pytest.fixture
def document_list():
    return [
        Document(
            page_content="Document 1",
            metadata={"source": "/path/to/first-document.txt"},
        ),
        Document(
            page_content="Document 2",
            metadata={"source": "/path/to/second_document.txt"},
        ),
        Document(
            page_content="Document 3",
            metadata={"source": "/path/to/third document.txt"},
        ),
    ]


@pytest.fixture
def long_document():
    return Document(
        page_content="This is a test document with a very long filename",
        metadata={
            "source": f"/path/to/{'very-long-document-name-that-exceeds-maximum-length-' * 3}.txt"
        },
    )


class TestMakeSafeSlug:
    @pytest.mark.parametrize(
        "input_text,expected",
        [
            ("Hello World", "hello-world"),
            ("Hello-World", "hello-world"),
            ("Hello_World", "hello_world"),
            ("Hello World!", "hello-world"),
            ("  Hello  World  ", "hello-world"),
            ("", ""),
            (None, ""),
            ("Hello--World", "hello-world"),
        ],
    )
    def test_make_safe_slug(self, input_text, expected):
        assert make_safe_slug(input_text) == expected


class TestGenerateSafeName:
    def test_generate_safe_name_from_chunk(self, sample_document):
        safe_name = generate_safe_name("default", chunk=sample_document)
        assert safe_name == "sample-document"

    def test_generate_safe_name_from_source(self):
        safe_name = generate_safe_name("default", source="/path/to/test-file.pdf")
        assert safe_name == "test-file"

    def test_generate_safe_name_fallback_to_prefix(self):
        safe_name = generate_safe_name("default-prefix")
        assert safe_name == "default-prefix"

    def test_generate_safe_name_chunk_without_source(self):
        doc = Document(page_content="No source", metadata={})
        safe_name = generate_safe_name("default-prefix", chunk=doc)
        assert safe_name == "default-prefix"

    def test_generate_safe_name_truncation(self, long_document):
        safe_name = generate_safe_name("default", chunk=long_document)
        assert len(safe_name) <= MAX_NAME_LENGTH
        assert safe_name.startswith("very-long-document-name-that-exceeds-maxim")
        assert len(safe_name) == TRUNCATED_NAME_LENGTH


class TestCreateChunkIds:
    def test_create_chunk_ids(self, document_list):
        chunk_ids = create_chunk_ids(document_list)
        assert len(chunk_ids) == len(document_list)
        for i, doc in enumerate(document_list):
            assert "id" in doc.metadata
            assert doc.metadata["id"] == chunk_ids[i]

    def test_create_chunk_ids_empty_list(self):
        chunk_ids = create_chunk_ids([])
        assert chunk_ids == []

    def test_create_chunk_ids_uniqueness(self, document_list):
        chunk_ids = create_chunk_ids(document_list)
        assert len(chunk_ids) == len(set(chunk_ids))  # All IDs should be unique

    def test_create_chunk_ids_with_prefix(self, document_list):
        prefix = "test-prefix"
        chunk_ids = create_chunk_ids(document_list, prefix=prefix)
        # If a document doesn't have source, it should use the prefix
        doc_without_source = Document(page_content="No source", metadata={})
        chunk_ids = create_chunk_ids([doc_without_source], prefix=prefix)
        assert chunk_ids[0].startswith(f"{prefix}-0-")

    def test_chunk_ids_format(self, document_list):
        chunk_ids = create_chunk_ids(document_list)
        for i, chunk_id in enumerate(chunk_ids):
            # Each ID should have format: name-index-uuid
            parts = chunk_id.split("-")
            assert parts[-2] == str(i)  # Check index
            assert len(parts[-1]) == UUID_SUFFIX_LENGTH  # Check UUID part length


class TestCreateImageId:
    def test_create_image_id(self):
        source = "/path/to/image.jpg"
        index = 5
        image_id = create_image_id(source, index)
        assert image_id.startswith("image-img5-")
        assert len(image_id.split("-")[-1]) == UUID_SUFFIX_LENGTH

    def test_create_image_id_with_prefix(self):
        source = ""  # Empty source
        index = 3
        prefix = "custom-prefix"
        image_id = create_image_id(source, index, prefix)
        assert image_id.startswith(f"{prefix}-img3-")

    def test_create_image_id_long_source(self):
        source = f"/path/to/{'very-long-image-name-' * 10}.jpg"
        index = 1
        image_id = create_image_id(source, index)
        name_part = image_id.split("-img1-")[0]
        assert len(name_part) <= TRUNCATED_NAME_LENGTH


class TestTextSplitterRecursiveChar:
    def test_text_splitter(self, document_list):
        # Create a document with longer content to ensure splitting
        long_text = "This is a test document. " * 100
        doc = Document(page_content=long_text, metadata={"source": "test.txt"})

        chunks = text_splitter_recursive_char([doc], chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

        # Check chunks are smaller than chunk_size
        for chunk in chunks:
            assert len(chunk.page_content) <= 100

    def test_text_splitter_small_document(self, sample_document):
        chunks = text_splitter_recursive_char(
            [sample_document], chunk_size=1000, chunk_overlap=0
        )
        assert len(chunks) == 1  # Document is small, should stay as a single chunk

    def test_text_splitter_empty_list(self):
        chunks = text_splitter_recursive_char([], chunk_size=1000, chunk_overlap=200)
        assert chunks == []

    def test_text_splitter_preserves_metadata(self, sample_document):
        chunks = text_splitter_recursive_char(
            [sample_document], chunk_size=10, chunk_overlap=0
        )
        for chunk in chunks:
            assert "source" in chunk.metadata
            assert chunk.metadata["source"] == sample_document.metadata["source"]
