import logging
from pathlib import Path
from typing import Self

from azure.storage.blob.aio import BlobServiceClient, ContainerClient
from fastapi import HTTPException

from src.configs.env_config import config
from src.services.storages.base_storage import BaseStorage

logger = logging.getLogger(__name__)


class BlobStorage(BaseStorage):
    """
    Service for retrieving documents in Azure Blob Storage.

    Attributes:
        blob_service_client (BlobServiceClient): The Azure BlobServiceClient instance.
        container_client (ContainerClient | None): The Azure ContainerClient instance.
    """

    def __init__(self) -> None:
        """
        Initialize the BlobStorage service with Azure BlobServiceClient.
        """
        super().__init__()
        self.blob_service_client: BlobServiceClient = (
            BlobServiceClient.from_connection_string(
                config.AZURE_STORAGE_CONNECTION_STRING
            )
        )
        self.container_client: ContainerClient = None

    async def __aenter__(self) -> Self:
        """
        Async context manager entry. Initializes the container client.
        Returns:
            Self: The BlobStorage instance.
        """
        await super().__aenter__()
        self.container_client = self.blob_service_client.get_container_client(
            container=config.AZURE_STORAGE_CONTAINER_NAME
        )
        return self

    async def download_blob(self, blob_name: str, temp_dir: str | Path) -> Path:
        """
        Download a blob from Azure Blob Storage to a temporary directory.

        Args:
            blob_name (str): The name of the blob to download.
            temp_dir (str | Path): The directory to save the downloaded blob.

        Returns:
            Path: The path to the downloaded blob file.

        Raises:
            HTTPException: If the blob is not found, empty, or download fails.
        """
        try:
            if not isinstance(temp_dir, Path):
                temp_dir = Path(temp_dir)

            # Get the blob client for the blob
            blob_client = self.container_client.get_blob_client(
                blob=f"chatbot/{blob_name}"
            )
            logger.debug(f"blob_client: {blob_client}")

            # Remove the timestamp from the blob name
            temp_pdf_path: Path = temp_dir / blob_name.split("/")[-1]

            with temp_pdf_path.open("wb") as download_file:
                logger.debug(f"Downloading blob to {temp_pdf_path}")

                stream = await blob_client.download_blob()
                data = await stream.readall()

                if not data:
                    logger.debug("No data downloaded from blob")
                    raise HTTPException(
                        status_code=404, detail="Blob not found or empty"
                    )

                download_file.write(data)

                logger.debug(f"Downloaded {len(data)} bytes from Azure blob")

                # Delete the blob after download
                await blob_client.delete_blob(delete_snapshots="include")
                logger.debug("Blob deleted successfully")

            if not temp_pdf_path.exists():
                logger.debug(f"Downloaded file does not exist: {temp_pdf_path}")
                raise HTTPException(status_code=404, detail="Blob not found or empty")

            return temp_pdf_path
        except Exception as e:
            logger.error(f"Error downloading blob: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download blob: {e}")

    async def close(self) -> None:
        """
        Close the BlobServiceClient and ContainerClient connections.
        """
        await self.container_client.close()
        await self.blob_service_client.close()
