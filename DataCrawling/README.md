# Data Crawling

<p align="left">
    <img alt="Selenium" src ="https://img.shields.io/badge/Selenium-343B02A.svg?&style=for-the-badge&logo=Selenium&logoColor=white"/>
</p>

### Environments

```bash
CHROMEDRIVER_PATH=****/****
OUTPUT_DIR=****
KEYWORD=고려대
RESTAURANT_FILENAME=****
IMAGE_FILENAME=****
REVIEW_FILENAME=****
```

I need a lot of reviews for restaurants.

I crawl some informations from [MangoPlate](https://www.mangoplate.com/). I see their `robots.txt` and I think that there are no problem to crawl their informations maybe..

## Steps

1. `crawl-restaurant.py` : Save a list of restaurant in `data/restaurant.json`.

2. `crawl-review.py` : Save a list of reviews in `data/review.json`.

3. `crawl-image.py` : Save a represantative image path for the restaurant in `data/image.json`.

