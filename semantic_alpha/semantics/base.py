from __future__ import annotations
from typing import Protocol
from ..schema import PostSemantics, SocialPost

class SemanticEngine(Protocol):
    async def classify(self, post: SocialPost) -> PostSemantics: ...
