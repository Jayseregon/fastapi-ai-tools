import logging

logger = logging.getLogger(__name__)


class SessionAdapter:
    """Adapter to make HttpClientService compatible with WebBaseLoader."""

    def __init__(self, client, cookies, headers, timeout=30.0):
        """Initialize SessionAdapter with explicit timeout."""
        self.client = client
        self.cookies = type("Cookies", (), {"get_dict": lambda self: cookies})()
        self.headers = headers
        self.verify = True  # SSL verification
        self.timeout = timeout
