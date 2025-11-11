from config import Config


class ExtractorConfig(Config):
    def __init__(self) -> None:
        super().__init__(
            {
                "GITHUB_AUTH_TOKEN": "auth_token",
                "GITHUB_BASE_URL": "base_url",
                "GITHUB_REPO_OWNER": "repo_owner",
                "GITHUB_REPO_NAME": "repo_name",
            }
        )

    def auth_token(self) -> str:
        return self["auth_token"]

    def base_url(self) -> str:
        return self["base_url"]

    def repo_owner(self) -> str:
        return self["repo_owner"]

    def repo_name(self) -> str:
        return self["repo_name"]
