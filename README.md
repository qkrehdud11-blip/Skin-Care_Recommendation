# SkinScope v7

피부 프로필과 고민 문장을 바탕으로 케어 방향과 성분을 안내하고, 유사한
올리브영 스킨케어 리뷰를 검색하는 Flask 애플리케이션입니다.

## 실행

```bash
.venv/bin/python web_app_v7.py
```

기본 주소는 `http://127.0.0.1:5004`입니다. 운영 환경에서는 관리자 계정과
세션 키를 환경 변수로 지정하세요.

```bash
export SKINSCOPE_SECRET_KEY='replace-with-a-random-secret'
export SKINSCOPE_ADMIN_USERNAME='admin'
export SKINSCOPE_ADMIN_PASSWORD='replace-with-a-strong-password'
export SKINSCOPE_ADMIN_EMAIL='admin@example.com'
.venv/bin/python web_app_v7.py
```

선택 가능한 실행 설정:

- `SKINSCOPE_HOST`: 기본값 `127.0.0.1`
- `SKINSCOPE_PORT`: 기본값 `5004`
- `SKINSCOPE_DEBUG`: `1`, `true`, `yes` 중 하나일 때만 디버그 모드 활성화

## 테스트

```bash
.venv/bin/python -m unittest -v test_web_app_v7.py
```

테스트는 임시 SQLite 데이터베이스를 사용하며 추천 API, 회원가입 검증,
인증 및 관리자 화면의 기본 동작을 확인합니다.

## v4_rebuild 데이터 및 알고리즘

v7 전문가 분석은 `web_app_v4_rebuild.py`와 같은 핵심 파이프라인을 사용합니다.

1. `datasets/skin_data_final.csv`의 `User Question`을 Okt로 형태소 분석
2. 명사·동사·형용사, 2글자 이상, `datasets/stopwords.csv` 제외
3. `models/tfidf_rebuild.pkl`로 입력을 TF-IDF 변환
4. `models/Tfidf_skin_data_rebuild.mtx`와 cosine similarity 계산
5. 입력한 프로필과 정확히 일치하는 후보군 안에서 고민 키워드 중첩을 보조 점수로 반영

현재 전문가 원본 데이터는 **문제성 피부 메이크업 추천 데이터**입니다. 따라서
`Makeup Response`에는 스킨케어 성분 안내뿐 아니라 파운데이션, 쿠션, 컨실러 등
메이크업 조언이 포함될 수 있습니다. 순수 스킨케어 처방 데이터로 해석하면 안 됩니다.

리뷰 모드는 다음 파일을 별도로 사용합니다.

- `datasets/oliveyoung_product_list.csv`
- `datasets/oliveyoung_reviews_preprocessed.csv`
- `models/tfidf_reviews.pkl`
- `models/Tfidf_reviews.mtx`

색조와 디바이스를 제외한 스킨/토너, 에센스/세럼/앰플, 크림, 로션 리뷰만
결과 후보로 사용합니다. 앱 시작 시 CSV 필수 열, 행렬 행 수, TF-IDF 특성 수가
일치하지 않으면 즉시 오류를 발생시킵니다.
