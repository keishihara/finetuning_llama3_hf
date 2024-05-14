import pathlib

from transformers import file_utils

PROJECT_HOME = pathlib.Path(__file__).parents[2]
PACKAGE_HOME = PROJECT_HOME / 'src' / 'tuner'
LOG_DIR = PROJECT_HOME / 'logs'

DEFAULT_CACHE_PATH = file_utils.default_cache_path

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = '[PAD]'
