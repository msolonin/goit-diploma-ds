from playwright.sync_api import sync_playwright
import os
import requests
import csv

boat_types = {
    "1": "MotorYacht",
    "2": "Seal",
    "3": "PowerBoat",
}

CURRENT_BOAT_TYPE = 1 # Can be 1, 2 or 3

URI = 'https://itboat.com'
URL = f"{URI}/explore/search?boats[type]={CURRENT_BOAT_TYPE}&boats[length_from]=&boats[length_to]=&boats[length_units]=m&boats[beam_from]=&boats[beam_to]=&boats[draft_from]=&boats[draft_to]=&boats[price_from]=&boats[price_to]=&boats[price_currency]=eur&boats[text]"
IMAGES_FOLDER = f'images{boat_types[CURRENT_BOAT_TYPE]}'
FILE_NAME = f'boats_{boat_types[CURRENT_BOAT_TYPE]}.csv'


class PageFetcher:
    def __init__(self, headless: bool = True, browser_type: str = "chromium"):
        self.headless = headless
        self.browser_type = browser_type
        self.playwright = None
        self.browser = None
        self.page = None

    def __enter__(self):
        self.playwright = sync_playwright().start()
        browser_launcher = getattr(self.playwright, self.browser_type)
        self.browser = browser_launcher.launch(headless=self.headless)
        self.page = self.browser.new_page()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.browser:
            self.browser.close()
        if self.playwright:
            self.playwright.stop()

    def get_page_source(self, url: str):
        self.page.goto(url)
        return self.page.content()

    def get_element(self, selector: str):
        return self.page.query_selector(selector)

    def get_elements(self, selector: str):
        return self.page.query_selector_all(selector)

    @staticmethod
    def get_attribute(element, attribute: str):
        return element.get_attribute(attribute)

    @staticmethod
    def save_image(url, folder, filename):
        os.makedirs(os.path.join(IMAGES_FOLDER, folder), exist_ok=True)
        path = os.path.join(IMAGES_FOLDER, folder, filename)
        if not os.path.exists(path):
            try:
                response = requests.get(url, stream=True)
                response.raise_for_status()
                with open(path, "wb") as f:
                    for chunk in response.iter_content(8192):
                        f.write(chunk)
                print(f" Saved image to: {path}")
            except Exception as e:
                print(f"Can't save image: {e}")

    @staticmethod
    def append_to_csv(file_path, data: dict):
        file_exists = os.path.isfile(file_path)
        with open(file_path, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)


if __name__ == "__main__":
    if not os.path.exists(IMAGES_FOLDER):
        os.mkdir(IMAGES_FOLDER)
    with PageFetcher(headless=False) as fetcher:
        html = fetcher.get_page_source(URL)
        paginator = fetcher.get_element(".pagination__items").inner_text().split("\n")
        paginator_int_values = [int(x) for x in paginator if isinstance(x, (int, str)) and str(x).isdigit()]
        min_value = min(paginator_int_values)
        max_value = max(paginator_int_values)
        for i in range(min_value, max_value + 1):
            page_url = URL + '=&page={}'.format(i)
            print(page_url)
            fetcher.get_page_source(page_url)
            items = fetcher.get_elements(".boat-item__ship")
            all_page_items = [{'href': f.get_attribute("href"),
                                'price': f.query_selector(".boat-item__price-main").inner_text(),
                                'boat_name': f.query_selector(".boat-item__name").inner_text(),
                                'boat_options': [f.inner_text() for f in f.query_selector_all(".boat-item__option")]}
                              for f in items]
            for item in all_page_items:
                boat_url = item['href']
                price = item['price']
                boat_name = item['boat_name']
                boat_options = item['boat_options']
                content = fetcher.get_page_source(boat_url)
                boat_type = fetcher.get_element(".shipyard-head__rubric").inner_text()
                boat_description = fetcher.get_element(".entity-boat__description").inner_text()
                try:
                    boat_review = fetcher.get_element(".entity-boat__rewiew").inner_text()
                except Exception as e:
                    print(e)
                    boat_review = None
                main_chars = [f.inner_text() for f in fetcher.get_elements(".shipyard-model__main-chars-item")]
                main_entity = [f.inner_text() for f in fetcher.get_elements(".entity-boat__table")]
                picture_urls = [f.get_attribute("href") for f in fetcher.get_elements(".gallery-model__link")]
                for picture_url in picture_urls:
                    picture_name = picture_url.split("/")[-1]
                    fetcher.save_image(URI + picture_url, boat_name, picture_name)
                fetcher.append_to_csv(FILE_NAME, {
                    "boat_name": boat_name,
                    "boat_type": boat_type,
                    "boat_description": boat_description,
                    "price": price,
                    "boat_options": "\n".join(boat_options),
                    "main_chars": "\n".join(main_chars),
                    "main_entity": "\n".join(main_entity),
                    "boat_url": boat_url,
                    "boat_review": boat_review
                })
