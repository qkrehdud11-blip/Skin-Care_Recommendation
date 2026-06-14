from playwright.sync_api import sync_playwright
import csv
import time
import os
import random
import re

# 🚀 [무제한 수집 모드] 모든 제품의 모든 리뷰 페이지를 끝까지 수집합니다.

def crawl_reviews_stealth():
    input_file = "./datasets/oliveyoung_product_list.csv"
    output_file = "./datasets/oliveyoung_reviews.csv"
    
    if not os.path.exists(input_file):
        print("❌ 제품 목록 파일이 없습니다. job02를 먼저 실행하세요.")
        return
        
    products = []
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        products = list(reader)
    
    # 이어받기: 이미 완료된 제품 체크
    processed_products = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                processed_products.add(row['product_name'])

    # 수집 대상: 아직 처리 안 된 모든 제품
    target_products = [p for p in products if p['product_name'] not in processed_products]
    
    if not target_products:
        print("✅ 모든 제품 수집이 이미 완료되었습니다.")
        return

    print(f"📦 총 {len(target_products)}개의 제품 수집을 시작합니다.")

    # 💡 브라우저 과부하 방지를 위해 20개 단위로 브라우저 재시작
    batch_size = 20
    for i in range(0, len(target_products), batch_size):
        batch = target_products[i:i + batch_size]
        print(f"\n🔄 브라우저 세션 시작 ({i + 1} ~ {min(i + batch_size, len(target_products))} 번째 제품)")
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False) # 로컬 모니터링 가능
            context = browser.new_context(
                user_agent="Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                viewport={'width': 1280, 'height': 1024}
            )
            page = context.new_page()
            page.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")

            with open(output_file, "a", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=["product_name", "star", "skin_type", "review"])
                if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                    writer.writeheader()

                for prod in batch:
                    print(f"\n👉 {prod['product_name']} 분석 중...")
                    try:
                        review_url = f"{prod['product_link']}&tab=review"
                        page.goto(review_url, wait_until="load", timeout=90000)
                        time.sleep(random.uniform(5, 8)) # 로딩 충분히 대기

                        # 리뷰 유무 확인 (총 건수 추출 시도)
                        total_count_text = "0"
                        try:
                            # Shadow DOM 내부의 총 건수 텍스트 확인
                            count_el = page.locator(".total-count").first
                            if count_el.is_visible():
                                total_count_text = count_el.inner_text()
                        except: pass
                        
                        if "0건" in total_count_text:
                            print("  ℹ️ 리뷰가 없는 상품입니다. 스킵.")
                            continue

                        # --- 페이지네이션 루프 (모든 페이지 수집) ---
                        p_idx = 1
                        while True:
                            # 스크롤을 내려서 렌더링 유도
                            page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                            time.sleep(2)
                            
                            # Shadow DOM 기반 리뷰 수집
                            items = page.locator("oy-review-review-item").all()
                            if not items:
                                print(f"    ⚠️ Page {p_idx}: 리뷰 아이템 로드 실패. (재시도 중...)")
                                time.sleep(3)
                                items = page.locator("oy-review-review-item").all()
                            
                            if not items: break

                            count = 0
                            for item in items:
                                try:
                                    # 리뷰 텍스트 (더보기 버튼이 있다면 클릭 시도할 수도 있지만 우선 보이는 것만)
                                    content = item.locator("oy-review-review-content p").inner_text(timeout=5000)
                                    
                                    # 피부 정보
                                    skin_info_elements = item.locator("oy-review-review-user .skin-type").all()
                                    skin_info = " / ".join([el.inner_text() for el in skin_info_elements])
                                    
                                    if len(content.strip()) > 10:
                                        writer.writerow({
                                            "product_name": prod['product_name'],
                                            "star": "5",
                                            "skin_type": skin_info,
                                            "review": content.strip()
                                        })
                                        count += 1
                                except: continue
                            
                            print(f"    Page {p_idx}: {count}개 리뷰 저장")

                            # 다음 페이지 버튼 찾기 및 클릭
                            try:
                                next_p = p_idx + 1
                                next_btn = page.locator(f"a[data-page-no='{next_p}']")
                                
                                # 만약 10페이지 단위로 넘어가는 화장표가 필요할 경우도 대비
                                if not next_btn.is_visible():
                                    # '다음' 화살표 버튼 탐색 (클래스나 텍스트로)
                                    next_arrow = page.locator("button[class*='next'], .pagination-next").first
                                    if next_arrow.is_visible():
                                        next_arrow.click()
                                        time.sleep(3)
                                        next_btn = page.locator(f"a[data-page-no='{next_p}']")
                                
                                if next_btn.is_visible():
                                    next_btn.click()
                                    p_idx += 1
                                    time.sleep(random.uniform(2, 4))
                                else:
                                    print("    🏁 마지막 페이지입니다.")
                                    break
                            except:
                                break
                                
                        f.flush() # 배치 단위 파일 저장
                    except Exception as e:
                        print(f"  ❌ 상품 처리 중 오류 (스킵): {e}")
                        continue
            
            browser.close()
            print(f"☕ 한 세트 완료. 차단 방지를 위해 10초 휴식...")
            time.sleep(10)

    print("\n✨ [임무 완료] 모든 카테고리 제품의 모든 리뷰를 수집했습니다!")

if __name__ == "__main__":
    crawl_reviews_stealth()
