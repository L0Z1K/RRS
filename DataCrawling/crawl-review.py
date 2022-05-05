"""
Crawl reviews of restaurant.
"""

import os
import requests
import json

from tqdm import tqdm

MAX_REVIEWS = 50

if __name__ == "__main__":
    OUTPUT_DIR = os.getenv("OUTPUT_DIR", "data")
    RESTAURANT_FILENAME = os.getenv("RESTAURANT_FILENAME", "restaurant.json")
    REVIEW_FILENAME = os.getenv("REVIEW_FILENAME", "review.json")

    with open(os.path.join(OUTPUT_DIR, RESTAURANT_FILENAME), "r") as f:
        restaurant = json.loads(f.read())

    review_dict = {}
    for key, value in tqdm(restaurant.items()):
        URL = f"https://stage.mangoplate.com/api/v5{value}/reviews.json?language=kor&device_uuid=RGcMs16516608776531272VyEIf&device_type=web&start_index=0&request_count={MAX_REVIEWS}&sort_by=2"
        res = requests.get(URL)

        reviews = json.loads(res.text)
        review_list = []
        for review in reviews:
            review_list.append(review["comment"]["comment"].strip())
        review_dict[key] = review_list

    with open(os.path.join(OUTPUT_DIR, REVIEW_FILENAME), "w") as f:
        f.write(json.dumps(review_dict, ensure_ascii=False, indent=2))
