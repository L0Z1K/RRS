# RRS: Restaurant Recommender System ๐ฝ

<p align="left">
    <img alt="Selenium" src ="https://img.shields.io/badge/Selenium-343B02A.svg?&style=for-the-badge&logo=Selenium&logoColor=white"/>
    <img alt="PyTorch Lightning" src="https://img.shields.io/badge/PyTorch Lightning-792EE5?style=for-the-badge&logo=PyTorch Lightning&logoColor=white" />
    <img alt="Streamlit" src ="https://img.shields.io/badge/Streamlit-FF4B4B.svg?&style=for-the-badge&logo=Streamlit&logoColor=white"/>
</p>

> Latte๋ ๋ง์ด์ผ.. ๋ฐฅ์ฝํ๋ฉด ์๋ด๊ธฐ๊ฐ ์๋น ๋ค ์กฐ์ฌํด์ ์๋๋ฐ ๋ง์ด์ผ.. ํ๋๋ ์กฐ์ฌ๋ฅผ ์ํด์์ด?? ์์.. ๋ญ ๋จน๊ณ  ์ถ์์ง ๋ฌธ์ฅ์ผ๋ก๋ง ์๊ธฐํด๋ด. AI๊ฐ ์ ์ผ ์ ํฉํ ๊ณณ์ผ๋ก ์ถ์ฒํด์ค๊ฒ!

This is a toy project for [DevKor](https://github.com/DevKor-Team) Mini-Web Hackathon ๐จโ๐ป.

๐จ I'm very bad at Web Development. So I focus on recommender system and my Streamlit did all of front-end.

My AI Model trained with a lot of reviews of restaurant near by Korea University.

When you give some query to my AI Model, it returns to you a list of restaurants you will love.

## Step

- Step1 : [Data Crawling](DataCrawling/)

- Step2 : [Modeling](Modeling/)

- Step3 : Inference with `server.py`.

```bash
$ streamlit run server.py
```

### Environments

```bash
NUM_LABELS=100
IMAGE_PATH="./DataCrawling/data/image.json"
RESTAURANT_PATH="./DataCrawling/data/restaurant.json"
CKPT_PATH=****
```

### Result

![Demo Gif](https://user-images.githubusercontent.com/64528476/166889432-f7136c7e-3209-4466-accf-af99c7dd9a98.gif)

