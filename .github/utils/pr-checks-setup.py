import sys
import os
import re
import json
import argparse
import requests

def main(args):
    with open(args.config_file, "r") as config_file:
        config = json.load(config_file)

    url = args.pr_url + "/files"
    heads = {
        "Authorization": "Bearer {}".format(os.getenv("GITHUB_TOKEN")),
        "X-GitHub-Api-Version": "2022-11-28",
       'Accept': 'application/vnd.github+json' }
    pr_info = json.loads(requests.get(url, headers=heads).text)

    # structure to store the checks to run and related information
    checks = {
        "flags": {
            "is_bom_change": False,
            "is_model_change": False,
            "is_container_change": False,
        },
        "files": {"pr": [], "bom": []},
        "dirs": {"workloads_to_run": set(), "containers_to_check": set()},
    }
    # directory structure for model or container dirs
    pipe = "|"
    valid_model_dir = re.compile(
        "^models_v2/"
        + f'({pipe.join(config["frameworks"])})/[\w-]+/({pipe.join(config["mode"])})/({pipe.join(config["platform"])})/[\w/.-]+$'
    )
    valid_container_dir = re.compile(
        "^docker/"
        + f'({pipe.join(config["frameworks"])})/[\w-]+/({pipe.join(config["mode"])})/({pipe.join(config["platform"])})/[\w/.-]+$'
    )

    for file in pr_info[:]:
        if file["status"] in ["added", "modified", "renamed", "copied", "changed"]:
            checks["files"]["pr"].append(f'{file["filename"]} [{file["status"]}]')
            # BoM changes
            if (
                file["filename"].split("/")[0] in ["models_v2", "docker"]
                and "requirements.txt" in file["filename"].split("/")[:]
            ):
                checks["flags"]["is_bom_change"] = True
                checks["files"]["bom"].append(file["filename"])
            # Model change
            if valid_model_dir.match(file["filename"]):
                model_root = "/".join(file["filename"].split("/")[0:5])
                container_root = "/".join(file["filename"].split("/")[0:5]).replace(
                    "models_v2", "docker"
                )
                checks["flags"]["is_model_change"] = True
                checks["dirs"]["workloads_to_run"].add(model_root)
                # Add check for dependent containers
                if os.path.exists(container_root):
                    checks["flags"]["is_container_change"] = True
                    checks["dirs"]["containers_to_check"].add(container_root)
            # Container change
            if valid_container_dir.match(file["filename"]):
                container_root = "/".join(file["filename"].split("/")[0:5])
                checks["flags"]["is_container_change"] = True
                checks["dirs"]["containers_to_check"].add(container_root)

    checks["dirs"]["workloads_to_run"] = list(checks["dirs"]["workloads_to_run"])
    checks["dirs"]["containers_to_check"] = list(checks["dirs"]["containers_to_check"])

    print(json.dumps(checks))


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
