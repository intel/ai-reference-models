import sys
import os
import re
import json
import yaml
import argparse

from termcolor import colored


def main(args):
    with open(args.config_file, "r") as config_file:
        config = json.load(config_file)

    dir_depth = re.compile(f"^{args.models_dir}(/[\w-]+){{4}}$")
    models_list = []
    succeded_check = True

    for root, dirs, files in os.walk(args.models_dir):
        if dir_depth.match(root):
            models_list.append(root[len(args.models_dir) + 1 :])

    print("Checking models:")
    for model_dir in models_list:
        print(f"\t{model_dir}")
        succeded_check = check_directory_structure(model_dir, config) and succeded_check
        succeded_check = (
            check_mandatory_files(args.models_dir, model_dir, config) and succeded_check
        )
        succeded_check = (
            check_test_structure(args.models_dir, model_dir, config, "tests.yaml")
            and succeded_check
        )
        print()

    if not succeded_check:
        sys.exit("Errors found!!! Please review logs")


def check_directory_structure(model_dir, config):
    pipe = "|"
    dir_structure = re.compile(
        f'({pipe.join(config["frameworks"])})/[\w-]+/({pipe.join(config["mode"])})/({pipe.join(config["platform"])})'
    )

    if dir_structure.match(model_dir):
        print("\t\tDirectory structure:", colored("OK", "green"))
        return True
    else:
        print("\t\tDirectory structure:", colored("INVALID", "red"))
        print(
            f'\n\t\t\tExpected: {pipe.join(config["frameworks"])}/<model_name>/{pipe.join(config["mode"])}/{pipe.join(config["platform"])}'
        )
        return False


def check_mandatory_files(root_dir, model_dir, config):
    if model_dir in config["skip_files_check"]:
        print("\t\tMandatory files:    ", colored("SKIPPED", "yellow"))
        return True
    existing_files = list(os.walk(f"{root_dir}/{model_dir}"))[0][2]
    missing_file = False
    for file in config["mandatory_files"]:
        if file not in existing_files:
            missing_file = True
            print(f"\t\t  Missing {file} in {model_dir}")
    if not missing_file:
        print("\t\tMandatory files:    ", colored("OK", "green"))
        return True
    else:
        print("\t\tMandatory files:    ", colored("MISSING", "red"))
        return False


def check_test_structure(root_dir, model_dir, config, test_file):
    if model_dir in config["skip_test_structure_check"]:
        print("\t\tTest structure:     ", colored("SKIPPED", "yellow"))
        return True

    if os.path.exists(f"{root_dir}/{model_dir}/{test_file}") and os.path.isfile(
        f"{root_dir}/{model_dir}/{test_file}"
    ):
        with open(f"{root_dir}/{model_dir}/{test_file}", "r") as yaml_file:
            tests = yaml.safe_load(yaml_file)

        valid_test_structure = True
        for test in list(tests):
            for key in config["test_keys"]:
                if key not in tests[test]:
                    valid_test_structure = False
                    print(f"Missing '{key}' key in {test} test at {test_file}")

        if valid_test_structure:
            print(f"\t\tTest structure:     ", colored("OK", "green"))
            return True
        else:
            print(f"\t\tTest structure:     ", colored("INVALID", "red"))
            return False
    else:
        print(f"\t\tTest structure:     ", colored("MISSING", "red"))
        return False


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(sys.argv)
    arg_parser.add_argument(
        "-d",
        "--models_dir",
        help="Directory path where models to check are.",
        required=True,
    )
    arg_parser.add_argument(
        "-c", "--config_file", help="Config file in YAML format.", required=True
    )
    args = arg_parser.parse_args()

    main(args)
