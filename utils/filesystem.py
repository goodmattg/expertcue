import argparse
import os
import operator
import sys
from functools import reduce


def path_exist(path):
    if os.path.isfile(path):
        if os.path.exists(path):
            return path
        else:
            raise argparse.ArgumentTypeError("File: '{0}' does not exist".format(path))
    else:
        if os.path.isdir(path):
            return path
        else:
            raise argparse.ArgumentTypeError("Path: '{0}' is invalid".format(path))


def file_exists(prospective_file):
    """Check if the prospective file exists"""
    file_path = os.path.join(
        os.getcwd(), os.path.basename(sys.argv[0]), prospective_file
    )
    if not os.path.exists(file_path):
        raise argparse.ArgumentTypeError("File: '{0}' does not exist".format(file_path))
    return file_path


def dir_exists_write_privileges(prospective_dir):
    """Check if the prospective directory exists with write privileges."""
    dir_path = os.path.join(os.getcwd(), os.path.basename(sys.argv[0]), prospective_dir)
    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' does not exist".format(dir_path)
        )
    elif not os.access(dir_path, os.W_OK):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' is not writable".format(dir_path)
        )
    return dir_path


def dir_exists_read_privileges(prospective_dir):
    """Check if the prospective directory exists with read privileges."""
    dir_path = os.path.join(os.getcwd(), os.path.basename(sys.argv[0]), prospective_dir)
    if not os.path.isdir(dir_path):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' does not exist".format(dir_path)
        )
    elif not os.access(dir_path, os.R_OK):
        raise argparse.ArgumentTypeError(
            "Directory: '{0}' is not readable".format(dir_path)
        )
    return dir_path
