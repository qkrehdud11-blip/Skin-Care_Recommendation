from playwright.sync_api import sync_playwright
import csv
import time
from bs4 import BeautifulSoup
import os

# 💡 올리브영 카테고리 매핑 (사용자 제공 및 표준 번호 기반)
CATEGORY_MAP = {
    # --- 스킨케어 하위 카테고리 ---
    "스킨/토너": "100000100010013",
    "에센스/세럼/앰플": "100000100010014",
    "크림": "100000100010015",
    "로션": "100000100010016",
    "미스트/오일": "100000100010011",
    "스킨케어 세트": "100000100010004",
    "스킨케어 디바이스": "100000100010010",
    
    # --- 메이크업 하위 카테고리 ---
    "립메이크업": "100000100020006",
    "베이스메이크업": "100000100020001",
    "아이메이크업": "100000100020007"
}

def get_product_list(playwright_page, cat_no, page):
    url = f"https://www.oliveyoung.co.kr/store/display/getMCategoryList.do?dispCatNo={cat_no}&pageIdx={page}&rowsPerPage=24"
    print(f"Fetching: Category {cat_no}, Page {page}")
    try:
        playwright_page.goto(url, wait_until="load", timeout=60000)
        time.sleep(3.5)
        return playwright_page.content()
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_product_list(html):
    if not html: return []
    soup = BeautifulSoup(html, 'lxml')
    items = soup.select(".prd_info")
    data = []
    for item in items:
        try:
            brand = item.select_one(".tx_brand").get_text(strip=True)
            name = item.select_one(".tx_name").get_text(strip=True)
            raw_link = item.select_one("a")["href"]
            link = raw_link if raw_link.startswith("http") else f"https://www.oliveyoung.co.kr{raw_link}"
            data.append({
                "product_brand": brand,
                "product_name": name,
                "product_link": link
            })
        except:
            continue
    return data

def crawl_products():
    PAGES_PER_CATEGORY = 2 # 테스트를 위해 일단 카테고리당 2페이지씩
    total_data = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36")
        page = context.new_page()
        
        for cat_name, cat_no in CATEGORY_MAP.items():
            print(f"\nProcessing Category: {cat_name}")
            for p_idx in range(1, PAGES_PER_CATEGORY + 1):
                html = get_product_list(page, cat_no, p_idx)
                items = parse_product_list(html)
                if not items: break
                for item in items:
                    item['category'] = cat_name
                total_data.extend(items)
                
        browser.close()
    
    os.makedirs("./datasets", exist_ok=True)
    with open("./datasets/oliveyoung_product_list.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "product_brand", "product_name", "product_link"])
        writer.writeheader()
        writer.writerows(total_data)
    print(f"\nFinished! Total products collected: {len(total_data)}")

if __name__ == "__main__":
    crawl_products()
