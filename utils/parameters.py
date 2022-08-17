import yaml
import argparse


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--params",
        required=True,
        help="yaml config file containing parameters required for model"
    )
    return parser


def get_params(params_file):
    with open(params_file, "r") as stream:
        params = yaml.safe_load(stream)
    return params