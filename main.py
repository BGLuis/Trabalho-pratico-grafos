from extractor.config import ExtractorConfig
from extractor.requests import fetch_all
from extractor.service import GithubService
from json import dumps


def main():
    config = ExtractorConfig()
    service = GithubService(config)
    print(dumps(fetch_all(service), indent=2))


if __name__ == "__main__":
    main()
