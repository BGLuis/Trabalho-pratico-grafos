from extractor.config import ExtractorConfig
from extractor.requests import fetch_all
from extractor.service import GithubService
from json import dump
from pathlib import Path


def main():
    config = ExtractorConfig()
    service = GithubService(config)

    with open(
        Path(__file__).parent / Path("data") / Path(f"{config.repo_name()}.json"), "w"
    ) as file:
        dump(fetch_all(service), file, indent=2)


if __name__ == "__main__":
    main()
