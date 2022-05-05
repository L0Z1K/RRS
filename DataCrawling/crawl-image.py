"""
Crawl a representative picture of the restaurant.
"""

import os
import json
import time

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import tqdm
from dotenv import load_dotenv

SELECTOR = "body > main > article > aside.restaurant-photos > div > div.owl-wrapper-outer > div > div:nth-child(1) > figure > figure > img"

if __name__ == "__main__":
    load_dotenv(verbose=True)

    CHROMEDRIVER_PATH = os.getenv("CROMEDRIVER_PATH", "")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data")
    RESTAURANT_FILENAME = os.getenv("RESTAURANT_FILENAME", "restaurant.json")
    IMAGE_FILENAME = os.getenv("IMAGE_FILENAME", "image.json")

    if CHROMEDRIVER_PATH == "":
        raise Exception("Please set the environment variable CHROMEDRIVER_PATH.")
    else:
        driver = webdriver.Chrome(CHROMEDRIVER_PATH)

    with open(os.path.join(OUTPUT_DIR, RESTAURANT_FILENAME), "r") as f:
        restaurant = json.loads(f.read())

    image_dict = {}
    for key, value in tqdm(restaurant.items()):
        driver.get(f"https://www.mangoplate.com{value}")
        time.sleep(3)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        notices = soup.select(SELECTOR)
        image_dict[key] = notices[0]["src"]

        with open(os.path.join(OUTPUT_DIR, IMAGE_FILENAME), "w") as f:
            f.write(json.dumps(image_dict, ensure_ascii=False, indent=2))
