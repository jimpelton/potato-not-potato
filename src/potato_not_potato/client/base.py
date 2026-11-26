from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profile import ModelProfile


__all__ = [
    "ModelClient"
]

class ModelClient(ABC):
    def __init__(self, profile: ModelProfile):
        self._profile = profile

    @property
    @abstractmethod
    def base_url(self) -> str: ...

    @abstractmethod
    def generate_structured_response(
        self,
        *,
        prompt: str,
        schema: dict,
        schema_name: str,
        image_mime: str,
        image_data: bytes|str,
    ) -> dict:
        """
        Generates a structured response from the model, using the profile to help generate the
        prompt and parse the structured response.

        :param prompt: The prompt to send to the model.
        :param schema: The JSON schema to validate the response.
        :param schema_name: The name of the schema.
        :param image_mime: the MIME type of the image.
        :param image_data: The image data of the receipt.
        """
        ...
