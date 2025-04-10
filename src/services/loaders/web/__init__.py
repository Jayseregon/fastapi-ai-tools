from src.services.loaders.web.public_loader import PublicLoader
from src.services.loaders.web.setics_loader import SeticsLoader
from src.services.loaders.web.web_image_loader import (
    WebImageLoader,
    create_web_image_loader,
)

__all__ = ["PublicLoader", "SeticsLoader", "WebImageLoader", "create_web_image_loader"]
