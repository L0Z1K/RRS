"""
Crawl a list of restaurant.
"""

import os
import json

from bs4 import BeautifulSoup
from selenium import webdriver
from tqdm import trange
from dotenv import load_dotenv

MAX_PAGES = 10

if __name__ == "__main__":
    load_dotenv(verbose=True)
    CHROMEDRIVER_PATH = os.getenv("CROMEDRIVER_PATH", "")
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data")
    KEYWORD = os.getenv("KEYWORD", "고려대")
    RESTAURANT_FILENAME = os.getenv("RESTAURANT_FILENAME", "restaurant.json")

    if CHROMEDRIVER_PATH == "":
        raise Exception("Please set the environment variable CHROMEDRIVER_PATH.")
    else:
        driver = webdriver.Chrome(CHROMEDRIVER_PATH)

    restaurant = {}
    for page in trange(1, MAX_PAGES + 1):
        URL = f"https://www.mangoplate.com/search/?keyword={KEYWORD}&page={page}"

        driver.get(URL)
        html = driver.page_source
        soup = BeautifulSoup(html, "html.parser")

        restaurants = soup.find_all("div", "info")
        for res in restaurants:
            try:
                title = res.select_one("h2").text.split("\n")[0].strip()
                href = res.select_one("a")["href"].strip()
                restaurant[title] = href
            except:
                pass

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(os.path.join(OUTPUT_DIR, RESTAURANT_FILENAME), "w") as f:
        f.write(json.dumps(restaurant, ensure_ascii=False, indent=2))
