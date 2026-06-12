import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from konlpy.tag import Okt
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent

# 서버 시작 시 데이터 및 모델 한 번만 로드
print("Loading data and models...")
df_skin = pd.read_csv(BASE_DIR / 'datasets' / 'skin_data_final.csv')
tfidf_matrix = mmread(BASE_DIR / 'models' / 'Tfidf_skin_data.mtx').tocsr()
with open(BASE_DIR / 'models' / 'tfidf.pkl', 'rb') as f:
    tfidf = pickle.load(f)
okt = Okt()

try:
    df_stopwords = pd.read_csv('../movie_review/datasets/stopwords.csv')
    stopwords = df_stopwords['stopword'].tolist()
except (FileNotFoundError, KeyError):
    stopwords = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게']

print("Load complete.")


def preprocess_input(text):
    text = re.sub('[^가-힣]', ' ', text)
    tokens = okt.pos(text, stem=True)
    words = [w for w, p in tokens if p in ['Noun', 'Verb', 'Adjective'] and len(w) > 1 and w not in stopwords]
    return ' '.join(words)


@app.route('/')
def index():
    return render_template('index_2.html')


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True) or {}
    user_input = data.get('keyword', '').strip()
    gender    = data.get('gender', '성별 선택')
    age       = data.get('age', '연령대 선택')
    skin_type = data.get('skin_type', '피부 타입 선택')

    # 키워드 없이 조건만 선택한 경우
    if not user_input:
        filtered = df_skin.copy()
        if gender != '성별 선택':
            filtered = filtered[filtered['Gender'] == gender]
        if age != '연령대 선택':
            filtered = filtered[filtered['Age'] == age]
        if skin_type != '피부 타입 선택':
            filtered = filtered[filtered['Skin Type'] == skin_type]
        if filtered.empty:
            return jsonify({'error': '조건에 맞는 데이터가 없습니다.'}), 404
        row = filtered.iloc[0]
        return _build_response(row, gender, age, skin_type)

    cleaned = preprocess_input(user_input)
    if not cleaned:
        return jsonify({'error': '입력하신 내용에서 유효한 단어를 찾을 수 없습니다.'}), 400

    # 전체 데이터 TF-IDF 유사도 계산
    input_vec = tfidf.transform([cleaned])
    cosine_sim = linear_kernel(input_vec, tfidf_matrix)[0]

    # 프로필 일치 보너스 (항목당 0.05, 최대 0.15)
    profile_bonus = np.zeros(len(df_skin))
    if gender != '성별 선택':
        profile_bonus += (df_skin['Gender'] == gender).values * 0.05
    if age != '연령대 선택':
        profile_bonus += (df_skin['Age'] == age).values * 0.05
    if skin_type != '피부 타입 선택':
        profile_bonus += (df_skin['Skin Type'] == skin_type).values * 0.05

    best_idx = int((cosine_sim + profile_bonus).argmax())
    row = df_skin.iloc[best_idx]
    return _build_response(row, gender, age, skin_type)


def _build_response(row, gender, age, skin_type):
    profile_parts = [v for v in [gender, age, skin_type]
                     if v not in ('성별 선택', '연령대 선택', '피부 타입 선택')]
    return jsonify({
        'profile':             ', '.join(profile_parts) if profile_parts else '전체',
        'response':            row['Makeup Response'],
        'rec_ingredients':     row['Recommended Ingredients'],
        'avoid_ingredients':   row['Ingredients to Avoid'],
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
