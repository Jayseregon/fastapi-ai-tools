from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException


# Patch BlobServiceClient.from_connection_string globally for all tests
@pytest.fixture(autouse=True)
def patch_blob_service_client():
    with patch("src.services.storages.blob_storage.BlobServiceClient") as mock_bsc:
        mock_bsc.from_connection_string.return_value = mock_bsc
        yield mock_bsc


from src.services.storages.blob_storage import BlobStorage


def test_blob_storage_init_sets_clients():
    instance = BlobStorage()
    assert hasattr(instance, "blob_service_client")
    assert instance.container_client is None


@pytest.mark.asyncio
async def test_blob_storage_aenter_sets_container_client(patch_blob_service_client):
    mock_container_client = AsyncMock()
    patch_blob_service_client.get_container_client.return_value = mock_container_client
    patch_blob_service_client.close = AsyncMock()

    storage = BlobStorage()
    async with storage as s:
        assert s.container_client == mock_container_client


@pytest.mark.asyncio
async def test_download_blob_success(patch_blob_service_client):
    storage = BlobStorage()
    storage.container_client = MagicMock()
    blob_client = MagicMock()
    storage.container_client.get_blob_client.return_value = blob_client

    temp_dir = Path("/tmp")
    blob_name = "test.pdf"
    temp_pdf_path = temp_dir / blob_name

    m_open = patch("pathlib.Path.open", MagicMock()).start()
    mock_file = MagicMock()
    m_open.return_value.__enter__.return_value = mock_file

    stream = AsyncMock()
    stream.readall.return_value = b"pdfdata"
    blob_client.download_blob = AsyncMock(return_value=stream)
    blob_client.delete_blob = AsyncMock()

    with patch.object(Path, "exists", return_value=True):
        result = await storage.download_blob(blob_name, temp_dir)
        assert result == temp_pdf_path
        blob_client.download_blob.assert_awaited_once()
        blob_client.delete_blob.assert_awaited_once()
        mock_file.write.assert_called_once_with(b"pdfdata")

    patch.stopall()


@pytest.mark.asyncio
async def test_download_blob_no_data(patch_blob_service_client):
    storage = BlobStorage()
    storage.container_client = MagicMock()
    blob_client = MagicMock()
    storage.container_client.get_blob_client.return_value = blob_client

    temp_dir = Path("/tmp")
    blob_name = "empty.pdf"

    m_open = patch("pathlib.Path.open", MagicMock()).start()
    mock_file = MagicMock()
    m_open.return_value.__enter__.return_value = mock_file

    stream = AsyncMock()
    stream.readall.return_value = b""
    blob_client.download_blob = AsyncMock(return_value=stream)
    blob_client.delete_blob = AsyncMock()

    with pytest.raises(HTTPException) as excinfo:
        with patch.object(Path, "exists", return_value=True):
            await storage.download_blob(blob_name, temp_dir)
    assert excinfo.value.status_code == 500
    assert "Failed to download blob" in str(excinfo.value.detail)

    patch.stopall()


@pytest.mark.asyncio
async def test_download_blob_file_not_exists(patch_blob_service_client):
    storage = BlobStorage()
    storage.container_client = MagicMock()
    blob_client = MagicMock()
    storage.container_client.get_blob_client.return_value = blob_client

    temp_dir = Path("/tmp")
    blob_name = "missing.pdf"

    m_open = patch("pathlib.Path.open", MagicMock()).start()
    mock_file = MagicMock()
    m_open.return_value.__enter__.return_value = mock_file

    stream = AsyncMock()
    stream.readall.return_value = b"pdfdata"
    blob_client.download_blob = AsyncMock(return_value=stream)
    blob_client.delete_blob = AsyncMock()

    with patch.object(Path, "exists", return_value=False):
        with pytest.raises(HTTPException) as excinfo:
            await storage.download_blob(blob_name, temp_dir)
    assert excinfo.value.status_code == 500
    assert "Failed to download blob" in str(excinfo.value.detail)

    patch.stopall()


@pytest.mark.asyncio
async def test_download_blob_raises_other(patch_blob_service_client):
    storage = BlobStorage()
    storage.container_client = MagicMock()
    blob_client = MagicMock()
    storage.container_client.get_blob_client.return_value = blob_client

    temp_dir = Path("/tmp")
    blob_name = "fail.pdf"

    _ = patch("pathlib.Path.open", MagicMock(side_effect=OSError("fail"))).start()

    with pytest.raises(HTTPException) as excinfo:
        await storage.download_blob(blob_name, temp_dir)
    assert excinfo.value.status_code == 500
    assert "Failed to download blob" in str(excinfo.value.detail)

    patch.stopall()


@pytest.mark.asyncio
async def test_blob_storage_close(patch_blob_service_client):
    storage = BlobStorage()
    storage.container_client = AsyncMock()
    storage.blob_service_client = AsyncMock()
    await storage.close()
    storage.container_client.close.assert_awaited_once()
    storage.blob_service_client.close.assert_awaited_once()
