import logging
from datetime import datetime
from typing import Dict, List, Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from langchain_community.document_loaders.parsers.images import LLMImageBlobParser
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI

from src.configs.env_config import config
from src.services.loaders.lib.http_client import HttpClient
from src.services.utils.embedding_toolkit import EmbeddingToolkit

logger = logging.getLogger(__name__)


class WebImageProcessor:
    """Handles crawling web pages and extracting images into Document objects."""

    def __init__(self, llm_model: Optional[BaseChatModel] = None):
        """
        Initialize the web image processor.

        Args:
            llm_model: Language model to use for image description
        """
        self.llm_model = llm_model or ChatOpenAI(
            model="gpt-4o-mini", api_key=config.OPENAI_API_KEY
        )
        self.image_parser = LLMImageBlobParser(model=self.llm_model)

    def _is_valid_image_url(self, url: str) -> bool:
        """
        Check if the URL is likely to be a valid image.

        Args:
            url: URL to check

        Returns:
            Boolean indicating if URL appears to be an image
        """
        if not url:
            return False

        # Check for common image extensions
        parsed = urlparse(url)
        path = parsed.path.lower()
        return path.endswith((".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp"))

    def _get_status_code(self, response) -> int:
        """
        Get status code from response object, handling different client implementations.

        Args:
            response: HTTP response object

        Returns:
            HTTP status code
        """
        if hasattr(response, "status_code"):
            return response.status_code
        elif hasattr(response, "status"):
            return response.status
        elif hasattr(response, "code"):
            return response.code
        else:
            logger.warning("Could not determine response status code")
            return 200

    async def _get_html_content(
        self, url: str, http_client: HttpClient
    ) -> Optional[str]:
        """
        Get HTML content from a URL.

        Args:
            url: URL to fetch
            http_client: HTTP client for requests

        Returns:
            HTML content as string or None if fetch fails
        """
        try:
            logger.debug(f"Fetching HTML content from: {url}")
            response = await http_client.client.get(url)

            # Check if the response has a status code
            status_code = self._get_status_code(response)

            if status_code != 200:
                logger.warning(f"Failed to fetch URL {url}: {status_code}")
                return None

            # Try to get the text content - handle different response types
            if hasattr(response, "text"):
                if callable(response.text):
                    try:
                        return await response.text()
                    except TypeError:
                        return response.text()
                else:
                    return response.text
            elif isinstance(response, str):
                return response
            elif hasattr(response, "content"):
                content = None
                if callable(response.content):
                    try:
                        content = await response.content()
                    except TypeError:
                        content = response.content()
                else:
                    content = response.content

                # Convert bytes to string if needed
                if isinstance(content, bytes):
                    return content.decode("utf-8", errors="replace")
                return content
            elif isinstance(response, bytes):
                return response.decode("utf-8", errors="replace")
            else:
                try:
                    return str(response)
                except Exception:
                    logger.warning(f"Could not convert response to string for {url}")
                    return None

        except Exception as e:
            logger.warning(f"Error fetching URL {url}: {str(e)}")
            return None

    def _extract_image_urls(
        self, html_content: str, page_url: Optional[str] = None
    ) -> List[Dict[str, str]]:
        """
        Extract image URLs and metadata from HTML content.

        Args:
            html_content: HTML content to parse
            page_url: Source page URL for resolving relative URLs and metadata

        Returns:
            List of dicts with image URLs and metadata
        """
        soup = BeautifulSoup(html_content, "html.parser")
        images = []

        # Process standard <img> tags
        for img in soup.find_all("img"):
            # Try different possible image URL attributes (handling lazy loading)
            img_url = None
            for attr in [
                "src",
                "data-src",
                "data-original",
                "data-lazy-src",
                "data-srcset",
            ]:
                if img.has_attr(attr) and img[attr]:
                    img_url = img[attr]
                    break

            # Skip if no URL found
            if not img_url:
                continue

            # Resolve relative URLs if page_url is provided
            if page_url and not img_url.startswith(("http://", "https://")):
                img_url = urljoin(page_url, img_url)

            # Collect metadata
            title = img.get("title", "")

            # Find any parent figure caption
            caption = ""
            parent_figure = img.find_parent("figure")
            if parent_figure:
                figcaption = parent_figure.find("figcaption")
                if figcaption:
                    caption = figcaption.get_text(strip=True)

            # Check for image captions in common patterns
            if not caption:
                # Look for adjacent caption elements
                next_sibling = img.find_next_sibling(
                    "span", class_="image-caption-text"
                )
                if next_sibling:
                    caption = next_sibling.get_text(strip=True)
                else:
                    # Look for parent div with caption class
                    parent_with_caption = img.find_parent("div", class_="image-caption")
                    if parent_with_caption:
                        caption_span = parent_with_caption.find(
                            "span", class_="image-caption-text"
                        )
                        if caption_span:
                            caption = caption_span.get_text(strip=True)

            # Check parent links for higher resolution versions
            parent_link = img.find_parent("a", href=True)
            if parent_link and self._is_valid_image_url(parent_link["href"]):
                high_res_url = parent_link["href"]
                if page_url and not high_res_url.startswith(("http://", "https://")):
                    high_res_url = urljoin(page_url, high_res_url)

                # Store both URLs for later filtering
                images.append(
                    {
                        "url": img_url,
                        "high_res_url": high_res_url,
                        "title": title,
                        "caption": caption,
                        "filename": self._extract_filename(img_url),
                    }
                )
            else:
                images.append(
                    {
                        "url": img_url,
                        "high_res_url": None,
                        "title": title,
                        "caption": caption,
                        "filename": self._extract_filename(img_url),
                    }
                )

        # Look for links to images (common for thumbnails)
        for link in soup.find_all("a", href=True):
            href = link["href"]
            # Check if the link points to an image file
            if href.lower().endswith(
                (".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp")
            ):
                # Resolve relative URLs if page_url is provided
                if page_url and not href.startswith(("http://", "https://")):
                    href = urljoin(page_url, href)

                # Skip if we've already processed this image
                already_included = any(
                    image["url"] == href or image["high_res_url"] == href
                    for image in images
                )
                if already_included:
                    continue

                # Use metadata from the nested image if available
                img = link.find("img")
                title = ""
                if img:
                    title = img.get("title", "")
                else:
                    # No nested image, use link title
                    title = link.get("title", "")

                # Check for captions
                caption = ""
                next_sibling = link.find_next_sibling(
                    "span", class_="image-caption-text"
                )
                if next_sibling:
                    caption = next_sibling.get_text(strip=True)

                images.append(
                    {
                        "url": href,
                        "high_res_url": None,
                        "title": title,
                        "caption": caption,
                        "filename": self._extract_filename(href),
                    }
                )

        return images

    def _extract_filename(self, url: str) -> str:
        """
        Extract the filename from a URL.

        Args:
            url: The URL to extract from

        Returns:
            The filename portion of the URL
        """
        parsed = urlparse(url)
        path = parsed.path
        return path.split("/")[-1]

    def _filter_setics_images(
        self, images: List[Dict[str, str]], page_url: str
    ) -> List[Dict[str, str]]:
        """
        Apply Setics-specific filtering to image URLs and enhance metadata.

        Args:
            images: List of image information dictionaries
            page_url: The URL of the page containing these images

        Returns:
            Filtered list of image information dictionaries with enhanced metadata
        """
        # First, filter out common UI elements
        filtered_images = []
        for img in images:
            # Skip UI elements and logos
            if any(
                pattern in img["url"].lower()
                for pattern in [
                    "logo-setics",
                    "vgrabber.png",
                    "favicon",
                    "icon-",
                    "button-",
                ]
            ):
                continue

            filtered_images.append(img)

        # Group images by filename to identify duplicates

        filename_groups: dict[str, list[dict]] = {}
        for img in filtered_images:
            if img["filename"] not in filename_groups:
                filename_groups[img["filename"]] = []
            filename_groups[img["filename"]].append(img)

        # Select the preferred version from each group
        deduplicated_images = []
        for filename, group in filename_groups.items():
            # If only one image with this filename, use it
            if len(group) == 1:
                deduplicated_images.append(group[0])
                continue

            # For multiple images with the same filename, prefer S3 URLs
            s3_versions = [img for img in group if "s3.amazonaws.com" in img["url"]]
            if s3_versions:
                # Use the Amazon S3 version
                deduplicated_images.append(s3_versions[0])
            else:
                # Use the first high-res version if available
                high_res_versions = [img for img in group if img["high_res_url"]]
                if high_res_versions:
                    # Use the high-res URL instead of the original
                    selected = high_res_versions[0].copy()
                    if selected["high_res_url"]:
                        selected["url"] = selected["high_res_url"]
                    deduplicated_images.append(selected)
                else:
                    # Just use the first one
                    deduplicated_images.append(group[0])

        # Enhance metadata for RAG
        enhanced_images = []
        timestamp = datetime.now().isoformat()

        img_id = EmbeddingToolkit.create_image_id(page_url)

        for img in deduplicated_images:
            # Create enhanced metadata dictionary
            enhanced_img = {
                "url": img["url"],
                "title": img["title"],
                "caption": img["caption"],
                "source": page_url,
                "document_type": "image",
                "timestamp": timestamp,
                "id": img_id,
            }
            enhanced_images.append(enhanced_img)

        return enhanced_images

    async def extract_setics_image_urls_from_url(
        self,
        url: str,
        http_client: HttpClient,
    ) -> List[Dict[str, str]]:
        """
        Extract and filter image URLs specifically for Setics website content.

        Args:
            url: URL of the webpage to extract images from
            http_client: HTTP client for making the request

        Returns:
            Filtered list of dictionaries containing image URLs and metadata
        """
        # Get HTML content
        html_content = await self._get_html_content(url, http_client)
        if not html_content:
            logger.warning(f"Could not fetch HTML content from {url}")
            return []

        # Extract all images
        all_images = self._extract_image_urls(html_content, url)

        # Apply Setics-specific filtering and enhance metadata
        filtered_images = self._filter_setics_images(all_images, url)

        # Log the results
        logger.debug(
            f"Found {len(all_images)} total images at {url}, filtered to {len(filtered_images)} relevant images"
        )

        return filtered_images
