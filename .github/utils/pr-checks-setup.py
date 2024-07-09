import sys
import os
import re
import json
import argparse
import requests
import hashlib

def main(args):
    with open(args.config_file, "r") as config_file:
        config = json.load(config_file)

    url = args.pr_url + "/files"
    heads = {
        "Authorization": "Bearer {}".format(os.getenv("GITHUB_TOKEN")),
        "X-GitHub-Api-Version": "2022-11-28",
       'Accept': 'application/vnd.github+json' }
    pr_info = json.loads(requests.get(url, headers=heads).text)
    
    print(json.dumps(build_checks_json(pr_info, config, url)))


def build_checks_json(pr_info, config, api_url):
    # structure to store the checks to run and related information
    checks_json = {
        "flags": {
            "is_bom_change": False,
            "is_model_change": False,
            "is_container_change": False,
        },
        "files": {"pr": [], "bom": []},
        "dirs": {"workloads_to_run": set(), "containers_to_check": set()},
    }
    
    PIPE = "|"

    url = api_url.replace('api.', '').replace('repos/', '').replace('pulls', 'pull')

    # directory structure for models dir
    valid_model_dir = re.compile(
        "^models_v2/"
        + f'({PIPE.join(config["frameworks"])})/[\w-]+/({PIPE.join(config["mode"])})/({PIPE.join(config["platform"])})/[\w/.-]+$'
    )

    # directory structure for containers dir
    valid_container_dir = re.compile(
        "^docker/"
        + f'({PIPE.join(config["frameworks"])})/[\w-]+/({PIPE.join(config["mode"])})/({PIPE.join(config["platform"])})/[\w/.-]+$'
    )

    # review changed files in the PR and activate check to run accordingly 
    for file in pr_info[:]:
        if file["status"] in ["added", "modified", "renamed", "copied", "changed"]:
            checks_json["files"]["pr"].append(f'{file["filename"]} [{file["status"]}]')
            # bom change
            if (
                file["filename"].split("/")[0] in ["models_v2", "docker"]
                and "requirements.txt" in file["filename"].split("/")[:]
            ):
                checks_json["flags"]["is_bom_change"] = True
                checks_json["files"]["bom"].append(build_filename_diff_url(file["filename"], url))
            # mmodel change
            if valid_model_dir.match(file["filename"]):
                model_root = "/".join(file["filename"].split("/")[0:5])
                container_root = "/".join(file["filename"].split("/")[0:5]).replace(
                    "models_v2", "docker"
                )
                checks_json["flags"]["is_model_change"] = True
                checks_json["dirs"]["workloads_to_run"].add(model_root)
                # add check for dependent containers
                if os.path.exists(container_root):
                    checks_json["flags"]["is_container_change"] = True
                    checks_json["dirs"]["containers_to_check"].add(container_root)
            # container change
            if valid_container_dir.match(file["filename"]):
                container_root = "/".join(file["filename"].split("/")[0:5])
                checks_json["flags"]["is_container_change"] = True
                checks_json["dirs"]["containers_to_check"].add(container_root)

    checks_json["dirs"]["workloads_to_run"] = list(checks_json["dirs"]["workloads_to_run"])
    checks_json["dirs"]["containers_to_check"] = list(checks_json["dirs"]["containers_to_check"])

    return checks_json


def build_filename_diff_url(filename, url): 
    file_info = {
        'filename': f'{filename}',
        'diff_url': ''
    }
    
    filename_hash = hashlib.sha256(filename.encode()).hexdigest()
    file_info['diff_url'] =  url + '#diff-' + filename_hash
    
    return file_info


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(sys.argv)

    arg_parser.add_argument(
        "-c", "--config_file", help="Config file in YAML format.", required=True
    )
    arg_parser.add_argument(
        "-u", "--pr_url", help="Pull request URL endpoint for REST calls", required=True
    )
    args = arg_parser.parse_args()

    main(args)
