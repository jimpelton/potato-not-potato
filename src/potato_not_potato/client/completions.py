"""
An open-ai compeltions API client.
"""

import base64
import json
import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, cast

from openai import OpenAI
from openai.types import ReasoningEffort
from openai.types.chat import (
    ChatCompletionContentPartImageParam,
    ChatCompletionContentPartParam,
    ChatCompletionContentPartTextParam,
    ChatCompletionMessage,
    ChatCompletionUserMessageParam,
)
from openai.types.shared_params import ResponseFormatJSONSchema

from .base import ModelClient

if TYPE_CHECKING:
    from .profile import ModelProfile


__all__ = [
    "CompletionsArguments",
    "CompletionsClient"
]


@dataclass
class CompletionsArguments:
    temperature: float = 0
    max_tokens: int = 4096
    top_p: float | None = None
    top_k: int | None = None
    presence_penalty: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be a finite nonnegative number")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if self.top_p is not None and (
            not math.isfinite(self.top_p) or not 0 <= self.top_p <= 1
        ):
            raise ValueError("top_p must be a finite number between 0 and 1")
        if self.top_k is not None and (type(self.top_k) is not int or self.top_k < 0):
            raise ValueError("top_k must be a nonnegative integer")
        if self.presence_penalty is not None and (
            not math.isfinite(self.presence_penalty)
            or not -2 <= self.presence_penalty <= 2
        ):
            raise ValueError(
                "presence_penalty must be a finite number between -2 and 2"
            )
        if self.seed is not None and (type(self.seed) is not int or self.seed < 0):
            raise ValueError("seed must be a nonnegative integer")

    def get_args(self) -> dict[str, int | float]:
        """Filter args that are None and return them as a dict."""
        return {
            name: value for name, value in asdict(self).items() if value is not None
        }


class CompletionsClient(ModelClient):
    def __init__(
        self,
        *,
        host: str,
        port: int,
        model_slug: str,
        profile: ModelProfile,
        args: CompletionsArguments | None = None,
    ) -> None:
        base_url = f"http://{host}:{port}/v1"
        self._model = model_slug
        self._args = args if args is not None else CompletionsArguments()
        self._client = OpenAI(
            base_url=base_url,
            api_key="__local__",
            max_retries=0,
        )
        super().__init__(profile)

    def base_url(self):
        return self.base_url

    def generate_structured_response(
        self,
        *,
        prompt: str,
        schema: dict[str, Any],
        schema_name: str,
        image_mime: str,
        image_data: bytes | str,
        timeout_s: int = 180,
    ) -> dict[str, object]:
        response_format: ResponseFormatJSONSchema = {
            "type": "json_schema",
            "json_schema": {
                "name": schema_name,
                "strict": True,
                "schema": schema,
            },
        }

        reply = self._client.chat.completions.create(
            model=self._model,
            messages=self.prepare_messages(prompt, image_mime, image_data),
            response_format=response_format,
            temperature=self._args.temperature,
            timeout=timeout_s,
            top_p=self._args.top_p,
            reasoning_effort="none",
        )

        try:
            choice = reply.choices[0]
            if choice.finish_reason == "length":
                raise ValueError(
                    f"answer truncated at the output/context token limit "
                    f"(requested max_tokens={self._args.max_tokens}); increase --max-tokens "
                    "or the server context length, or use a smaller input"
                )
            text = self.extract_answer(choice.message)
            data: object = json.loads(text)
            if not isinstance(data, dict):
                raise TypeError(f"expected a JSON object, got {type(data)}")

            # The JSON decoder guarantees that object keys are strings.
            return cast(dict[str, object], data)
        except (ValueError, KeyError, IndexError, TypeError, AttributeError) as error:
            raise ValueError(f"unusable model response: {error}") from error

    def prepare_messages(
        self, prompt: str, image_mime: str, image_data: bytes | str
    ) -> list[ChatCompletionUserMessageParam]:
        """
        Prepares the messages for the model, assuming a chat completions api.

        :param image_mime: the mime type of the image contained in image_data.
        :param prompt: The prompt to send to the model.
        :param image_data: the image, if str assumes already in base64.
        :return: The prepared list of user messages.
        """

        text: ChatCompletionContentPartTextParam = {"type": "text", "text": prompt}

        if isinstance(image_data, bytes):
            image_data = base64.b64encode(image_data).decode("ascii")
        image_url = f"data:{image_mime};base64,{image_data}"

        image: ChatCompletionContentPartImageParam = {
            "type": "image_url",
            "image_url": {"url": image_url},
        }
        content: list[ChatCompletionContentPartParam]
        if self._profile.image_first:
            content = [image, text]
        else:
            content = [text, image]

        return [
            {
                "role": "user",
                "content": content,
            }
        ]

    def extract_answer(self, message: ChatCompletionMessage) -> str:
        """
        Extracts the answer from the model response.

        :param message: The model response message (json).
        :return: The extracted answer as a string.
        """
        content = message.content
        if content is None:
            return ""

        if not isinstance(content, str):
            raise TypeError("expected textual answer content")

        text = self._profile.strip_reasoning(content.strip())
        return text
