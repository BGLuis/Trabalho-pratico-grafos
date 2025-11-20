import pandas as pd
import json


# def main():
#     print("Hello from trabalho-pratico-grafos!")
#     with open("starship.json", "r", encoding="utf-8") as f:
#         data = json.load(f)

#     # Converte pullRequests em CSV
#     if "pullRequests" in data:
#         df_pr = pd.DataFrame(data["pullRequests"])
#         df_pr.to_csv("pullRequests.csv", index=False, encoding="utf-8")

#         # df = pd.DataFrame(
#         #     {
#         #         "title": pr["title"],
#         #         "createdAt": pr["createdAt"],
#         #         "mergedAt": pr["mergedAt"],
#         #         "author": pr["author"]["login"] if pr["author"] else None,
#         #         "additions": pr["additions"],
#         #         "deletions": pr["deletions"],
#         #         "changedFiles": pr["changedFiles"],
#         #     }
#         #     for pr in data["pullRequests"]
#         # )

#     # Converte issues em CSV
#     if "issues" in data:
#         df_issues = pd.DataFrame(data["issues"])
#         df_issues.to_csv("issues.csv", index=False, encoding="utf-8")

#         df = pd.DataFrame(
#             {
#                 "author": issue["author"]["login"] if issue["author"] else None,
#                 "comments": issue["comments"]["author"] if issue["comments"] else None,
#             }
#             for issue in data["issues"]
#         )

#         df.to_csv("issues_relacoes.csv", index=False, encoding="utf-8")

#     print("Arquivos CSV gerados com sucesso!")


def extract_authors_from_issues(data):
    issues = data.get("issues", [])
    extracted_data = []

    for issue in issues:
        issue_author = issue["author"]["login"] if issue.get("author") else None
        comment_authors = [
            comment["author"]["login"]
            for comment in issue.get("comments", [])
            if comment.get("author")
        ]
        extracted_data.append(
            {
                "issue_author": issue_author,
                "comment_authors": ", ".join(comment_authors),
            }
        )

    df = pd.DataFrame(extracted_data)
    df.to_csv("issues_authors.csv", index=False, encoding="utf-8")


def extract_closed_issues(data):
    issues = data.get("issues", [])
    extracted_data = []
    for issue in issues:
        issue_author = issue["author"]["login"] if issue.get("author") else None
        closed_by = None
        if issue.get("timelineItems"):
            for item in issue["timelineItems"]:
                closed_by = item["actor"]["login"] if item.get("actor") else None

        extracted_data.append(
            {
                "issue_author": issue_author,
                "closed_by": closed_by,
            }
        )

    df = pd.DataFrame(extracted_data)
    df.to_csv("closed_issues.csv", index=False, encoding="utf-8")


if __name__ == "__main__":
    with open("starship.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    extract_authors_from_issues(data)
    extract_closed_issues(data)
