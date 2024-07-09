import sys
import os
import json
import argparse
import requests


def main(args):
    pr_url = args.pr_url
    bom_files_json = args.bom_files_json

    bom_change_comment_body = build_bom_change_comment_body(bom_files_json)
    bom_change_comment_id = find_bom_change_comment(pr_url)

    if bom_change_comment_id == None:
        post_bom_change_comment(pr_url, bom_change_comment_body)
    else:
        update_bom_change_comment(
            pr_url, bom_change_comment_id, bom_change_comment_body
        )


def find_bom_change_comment(pr_url):
    comments_url = pr_url.replace("pulls", "issues") + "/comments"
    heads = {
        "Authorization": "Bearer {}".format(os.getenv("GITHUB_TOKEN")),
        "X-GitHub-Api-Version": "2022-11-28",
        "Accept": "application/vnd.github+json",
    }
    pr_comments = json.loads(requests.get(comments_url, headers=heads).text)
    bom_change_comment_id = None
    for comment in pr_comments:
        if comment["body"].find("# BoM change found") != -1:
            bom_change_comment_id = comment["id"]
    return bom_change_comment_id


def post_bom_change_comment(pr_url, bom_change_comment_body):
    comments_url = pr_url.replace("pulls", "issues") + "/comments"
    heads = {
        "Authorization": "Bearer {}".format(os.getenv("GITHUB_TOKEN")),
        "X-GitHub-Api-Version": "2022-11-28",
        "Accept": "application/vnd.github+json",
    }

    body = {"body": bom_change_comment_body}

    response = requests.post(comments_url, headers=heads, json=body)
    print(response.text)


def update_bom_change_comment(pr_url, bom_change_comment_id, bom_change_comment_body):
    SLASH = "/"
    pr_url_numberless = SLASH.join(pr_url.split(SLASH)[:-1])
    comment_url = (
        pr_url_numberless.replace("pulls", "issues")
        + "/comments"
        + "/{}".format(bom_change_comment_id)
    )
    heads = {
        "Authorization": "Bearer {}".format(os.getenv("GITHUB_TOKEN")),
        "X-GitHub-Api-Version": "2022-11-28",
        "Accept": "application/vnd.github+json",
    }

    body = {"body": bom_change_comment_body}

    response = requests.patch(comment_url, headers=heads, json=body)
    print(response.text)


def build_bom_change_comment_body(bom_files_json_input):
    bom_files_json = json.loads(bom_files_json_input)
    bom_change_comment_body = "# BoM change found\r\n## List of files\r\n|   |\r\n|:--|"

    for file in bom_files_json:
        bom_change_comment_body += "\r\n[{}]({})".format(
            file["filename"], file["diff_url"]
        )

    return bom_change_comment_body


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(sys.argv)

    arg_parser.add_argument(
        "-u", "--pr_url", help="Pull request URL endpoint for REST calls", required=True
    )
    arg_parser.add_argument(
        "-j", "--bom_files_json", help="JSON structure with checks information", required=True
    )
    args = arg_parser.parse_args()

    main(args)
