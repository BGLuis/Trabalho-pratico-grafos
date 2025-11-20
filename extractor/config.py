from config import Config
from typing import Dict, List, Optional
from datetime import datetime, timezone

from utils import log


class TokenManager:
    """Manages multiple GitHub tokens and handles rotation when rate limited."""

    def __init__(self, tokens: List[str]):
        if not tokens:
            raise ValueError("At least one token must be provided")
        self.__tokens = tokens
        self.__current_index = 0
        self.__rate_limits: Dict[str, Optional[datetime]] = {
            token: None for token in tokens
        }
        log(f"Initialized TokenManager with {len(tokens)} token(s)")

    def get_current_token(self) -> str:
        return self.__tokens[self.__current_index]

    def mark_rate_limited(self, token: str, reset_at: Optional[datetime]) -> None:
        self.__rate_limits[token] = reset_at
        log(
            f"Token {self.__tokens.index(token) + 1}/{len(self.__tokens)} marked as rate limited until {reset_at}"
        )

    def get_next_available_token(self) -> tuple[str, Optional[datetime]]:
        now = datetime.now(timezone.utc)

        for i in range(len(self.__tokens)):
            idx = (self.__current_index + i) % len(self.__tokens)
            token = self.__tokens[idx]
            reset_at = self.__rate_limits[token]

            if reset_at is None or reset_at <= now:
                self.__current_index = idx
                self.__rate_limits[token] = None  # Clear the rate limit
                log(f"Switched to token {idx + 1}/{len(self.__tokens)}")
                return (token, None)

        min_wait_token = None
        min_reset_at = None
        min_wait_seconds = float("inf")

        for token, reset_at in self.__rate_limits.items():
            if reset_at is not None:
                wait_seconds = (reset_at - now).total_seconds()
                if wait_seconds < min_wait_seconds:
                    min_wait_seconds = wait_seconds
                    min_wait_token = token
                    min_reset_at = reset_at

        if min_wait_token:
            self.__current_index = self.__tokens.index(min_wait_token)
            log(
                f"All tokens rate limited. Using token {self.__current_index + 1}/{len(self.__tokens)} with shortest wait ({min_wait_seconds:.0f}s)"
            )
            return (min_wait_token, min_reset_at)

        return (self.get_current_token(), None)


class ExtractorConfig(Config):
    def __init__(self) -> None:
        super().__init__(
            {
                "GITHUB_AUTH_TOKENS": "auth_tokens",
                "GITHUB_BASE_URL": "base_url",
                "GITHUB_REPO_OWNER": "repo_owner",
                "GITHUB_REPO_NAME": "repo_name",
            }
        )

    def auth_tokens(self) -> List[str]:
        """Returns a list of auth tokens (comma-separated)."""
        tokens_str = self["auth_tokens"]
        tokens = [token.strip() for token in tokens_str.split(",")]
        return [token for token in tokens if token]

    def base_url(self) -> str:
        return self["base_url"]

    def repo_owner(self) -> str:
        return self["repo_owner"]

    def repo_name(self) -> str:
        return self["repo_name"]
