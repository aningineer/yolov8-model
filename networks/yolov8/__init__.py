# Ultralytics YOLO 🚀, GPL-3.0 license


import contextlib
import inspect
import logging.config
import os
import platform
import re
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Union

import cv2
import numpy as np
import pandas as pd
import torch
import yaml

from networks.yolov8.custom import LOGGER


FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]  # YOLO
AUTOINSTALL = str(os.getenv('YOLO_AUTOINSTALL', True)).lower() == 'true'  # global auto-install mode
RANK = int(os.getenv('RANK', -1))

def is_dir_writeable(dir_path: Union[str, Path]) -> bool:
    """
    Check if a directory is writeable.

    Args:
        dir_path (str) or (Path): The path to the directory.

    Returns:
        bool: True if the directory is writeable, False otherwise.
    """
    try:
        with tempfile.TemporaryFile(dir=dir_path):
            pass
        return True
    except OSError:
        return False


def get_user_config_dir(sub_dir='Ultralytics'):
    """
    Get the user config directory.

    Args:
        sub_dir (str): The name of the subdirectory to create.

    Returns:
        Path: The path to the user config directory.
    """
    # Get the operating system name
    os_name = platform.system()

    # Return the appropriate config directory for each operating system
    if os_name == 'Windows':
        path = Path.home() / 'AppData' / 'Roaming' / sub_dir
    elif os_name == 'Darwin':  # macOS
        path = Path.home() / 'Library' / 'Application Support' / sub_dir
    elif os_name == 'Linux':
        path = Path.home() / '.config' / sub_dir
    else:
        raise ValueError(f'Unsupported operating system: {os_name}')

    # GCP and AWS lambda fix, only /tmp is writeable
    if not is_dir_writeable(str(path.parent)):
        path = Path('/tmp') / sub_dir

    # Create the subdirectory if it does not exist
    path.mkdir(parents=True, exist_ok=True)

    return path


USER_CONFIG_DIR = get_user_config_dir()  # Ultralytics settings dir


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        # Add YAML filename to dict and return
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def emojis(string=''):
    # Return platform-dependent emoji-safe version of string
    return string.encode().decode('ascii', 'ignore') if platform.system() == 'Windows' else string


class TryExcept(contextlib.ContextDecorator):
    # YOLOv8 TryExcept class. Usage: @TryExcept() decorator or 'with TryExcept():' context manager
    def __init__(self, msg='', verbose=True):
        self.msg = msg
        self.verbose = verbose

    def __enter__(self):
        pass

    def __exit__(self, exc_type, value, traceback):
        if self.verbose and value:
            print(emojis(f"{self.msg}{': ' if self.msg else ''}{value}"))
        return True

def colorstr(*input):
    # Colors a string https://en.wikipedia.org/wiki/ANSI_escape_code, i.e.  colorstr('blue', 'hello world')
    *args, string = input if len(input) > 1 else ("blue", "bold", input[0])  # color arguments, string
    colors = {
        "black": "\033[30m",  # basic colors
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "white": "\033[37m",
        "bright_black": "\033[90m",  # bright colors
        "bright_red": "\033[91m",
        "bright_green": "\033[92m",
        "bright_yellow": "\033[93m",
        "bright_blue": "\033[94m",
        "bright_magenta": "\033[95m",
        "bright_cyan": "\033[96m",
        "bright_white": "\033[97m",
        "end": "\033[0m",  # misc
        "bold": "\033[1m",
        "underline": "\033[4m"}
    return "".join(colors[x] for x in args) + f"{string}" + colors["end"]



def yaml_save(file='data.yaml', data=None):
    """
    Save YAML data to a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        data (dict, optional): Data to save in YAML format. Default is None.

    Returns:
        None: Data is saved to the specified file.
    """
    file = Path(file)
    if not file.parent.exists():
        # Create parent directories if they don't exist
        file.parent.mkdir(parents=True, exist_ok=True)

    with open(file, 'w') as f:
        # Dump data to file in YAML format, converting Path objects to strings
        yaml.safe_dump({k: str(v) if isinstance(v, Path) else v for k, v in data.items()}, f, sort_keys=False)


def yaml_load(file='data.yaml', append_filename=False):
    """
    Load YAML data from a file.

    Args:
        file (str, optional): File name. Default is 'data.yaml'.
        append_filename (bool): Add the YAML filename to the YAML dictionary. Default is False.

    Returns:
        dict: YAML data and file name.
    """
    with open(file, errors='ignore', encoding='utf-8') as f:
        # Add YAML filename to dict and return
        s = f.read()  # string
        if not s.isprintable():  # remove special characters
            s = re.sub(r'[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+', '', s)
        return {**yaml.safe_load(s), 'yaml_file': str(file)} if append_filename else yaml.safe_load(s)


def yaml_print(yaml_file: Union[str, Path, dict]) -> None:
    """
    Pretty prints a yaml file or a yaml-formatted dictionary.

    Args:
        yaml_file: The file path of the yaml file or a yaml-formatted dictionary.

    Returns:
        None
    """
    yaml_dict = yaml_load(yaml_file) if isinstance(yaml_file, (str, Path)) else yaml_file
    dump = yaml.dump(yaml_dict, default_flow_style=False)
    LOGGER.info(f"Printing '{colorstr('bold', 'black', yaml_file)}'\n\n{dump}")

def get_git_dir():
    """
    Determines whether the current file is part of a git repository and if so, returns the repository root directory.
    If the current file is not part of a git repository, returns None.

    Returns:
        (Path) or (None): Git root directory if found or None if not found.
    """
    for d in Path(__file__).parents:
        if (d / '.git').is_dir():
            return d
    return None  # no .git dir found


def is_colab():
    """
    Check if the current script is running inside a Google Colab notebook.

    Returns:
        bool: True if running inside a Colab notebook, False otherwise.
    """
    return 'COLAB_RELEASE_TAG' in os.environ or 'COLAB_BACKEND_VERSION' in os.environ


def is_kaggle():
    """
    Check if the current script is running inside a Kaggle kernel.

    Returns:
        bool: True if running inside a Kaggle kernel, False otherwise.
    """
    return os.environ.get('PWD') == '/kaggle/working' and os.environ.get('KAGGLE_URL_BASE') == 'https://www.kaggle.com'


def is_jupyter():
    """
    Check if the current script is running inside a Jupyter Notebook.
    Verified on Colab, Jupyterlab, Kaggle, Paperspace.

    Returns:
        bool: True if running inside a Jupyter Notebook, False otherwise.
    """
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False

def get_settings(file=USER_CONFIG_DIR / 'settings.yaml', version='0.0.1'):
    """
    Loads a global Ultralytics settings YAML file or creates one with default values if it does not exist.

    Args:
        file (Path): Path to the Ultralytics settings YAML file. Defaults to 'settings.yaml' in the USER_CONFIG_DIR.
        version (str): Settings version. If min settings version not met, new default settings will be saved.

    Returns:
        dict: Dictionary of settings key-value pairs.
    """

    from networks.yolov8.checks import check_version
    from networks.yolov8.torch_utils import torch_distributed_zero_first
    
    git_dir = get_git_dir()
    root = git_dir or Path()
    datasets_root = (root.parent if git_dir and is_dir_writeable(root.parent) else root).resolve()
    defaults = {
        'datasets_dir': str(datasets_root / 'datasets'),  # default datasets directory.
        'weights_dir': str(root / 'weights'),  # default weights directory.
        'runs_dir': str(root / 'runs'),  # default runs directory.
        'sync': True,  # sync analytics to help with YOLO development
        'uuid': uuid.getnode(),  # device UUID to align analytics
        'settings_version': version}  # Ultralytics settings version

    with torch_distributed_zero_first(RANK):
        if not file.exists():
            yaml_save(file, defaults)
        settings = yaml_load(file)

        # Check that settings keys and types match defaults
        correct = \
            settings.keys() == defaults.keys() \
            and all(type(a) == type(b) for a, b in zip(settings.values(), defaults.values())) 
            # and check_version(settings['settings_version'], version)
        if not correct:
            LOGGER.warning('WARNING ⚠️ Ultralytics settings reset to defaults. '
                           '\nThis is normal and may be due to a recent ultralytics package update, '
                           'but may have overwritten previous settings. '
                           f"\nYou may view and update settings directly in '{file}'")
            settings = defaults  # merge **defaults with **settings (prefer **settings)
            yaml_save(file, settings)  # save updated defaults

        return settings


SETTINGS = get_settings()