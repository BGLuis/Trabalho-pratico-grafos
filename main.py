from extractor.config import ExtractorConfig
from extractor.requests import fetch_all
from extractor.service import GithubService


def main():
    config = ExtractorConfig()
    service = GithubService(config)
    print(fetch_all(service))


if __name__ == "__main__":
    main()
