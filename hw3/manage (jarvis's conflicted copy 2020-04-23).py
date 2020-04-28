import os
from pathlib import Path

ROOT_DIR = Path(f"{os.getcwd()}").parent
DATA_DIR = f"{ROOT_DIR}/data"
IMAGE_DIR = f"{ROOT_DIR}/images"
NPS_DATA_DIR = f"{DATA_DIR}/nps_chat"  # nps chat dataset
COVID_REDDIT_DATA_DIR = f"{DATA_DIR}/covid_reddit_posts"  # COVID-19 related reddit post dataset
