from pathlib import Path

from transformers import file_utils

PROJECT_HOME = Path(__file__).parents[2]
LOG_DIR = PROJECT_HOME / 'logs'

PACKAGE_HOME = Path(__file__).parent

DEFAULT_CACHE_PATH = file_utils.default_cache_path

IGNORE_INDEX = -100

DEFAULT_PAD_TOKEN = '[PAD]'
