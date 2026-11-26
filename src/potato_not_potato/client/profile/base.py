from dataclasses import dataclass

__all__ = ["Gemma4Profile", "ModelProfile"]

@dataclass(frozen=True)
class ModelProfile:
    reasoning_delimiters: tuple[str, str] | None = None
    image_first: bool = True

    def strip_reasoning(self, text: str) -> str:
        if self.reasoning_delimiters:
            opening, closing = self.reasoning_delimiters

            # this loop handles multiple consecutive reasoning blocks
            while text.startswith(opening):
                # separator ends up being the closing tag
                # text is everything after the closing tag
                _, separator, text = text[len(opening) :].partition(closing)
                # A truncated thought block must never be mistaken for the answer.
                if not separator:
                    raise ValueError("incomplete reasoning block")
                text = text.lstrip()
        return text


Gemma4Profile = ModelProfile(
    reasoning_delimiters=("<|channel>thought", "<channel|>"),
    image_first=True,
)

