from pydantic import BaseModel


class User(BaseModel):
    """
    Represents a user authenticated via JWT.

    Attributes:
        id (str): The unique identifier of the user.
        email (str): The email address of the user.
        name (str): The name of the user.
        issuer (str): The issuer of the JWT.
        issued_at (int): The timestamp when the JWT was issued.
        expires_at (int): The timestamp when the JWT expires.
    """

    id: str
    email: str
    name: str
    issuer: str
    issued_at: int
    expires_at: int
