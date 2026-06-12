import numpy as np
import sys
import os
import pandas as pd
import pickle
import re
from konlpy.tag import Okt
from sklearn.metrics.pairwise import linear_kernel
from scipy.io import mmread
from PyQt5.QtWidgets import QApplication, QWidget, QMessageBox
from PyQt5 import uic

# UI 파일 로드
form_window = uic.loadUiType('./skincare_recommendation_final.ui')[0]

class SkincareRecommendationApp(QWidget, form_window):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 데이터 및 모델 로드
        print("Loading data and models...")
        try:
            self.df_skin = pd.read_csv('./datasets/skin_data_final.csv')
            self.tfidf_matrix = mmread('./models/Tfidf_skin_data.mtx').tocsr()
            with open('./models/tfidf.pkl', 'rb') as f:
                self.tfidf = pickle.load(f)
            print("Load complete.")
        except Exception as e:
            print(f"Error loading files: {e}")
            QMessageBox.critical(self, "오류", "데이터 파일을 로드하는 중 오류가 발생했습니다.")

        self.okt = Okt()
        
        # 불용어 로드
        try:
            df_stopwords = pd.read_csv('../movie_review/datasets/stopwords.csv')
            self.stopwords = df_stopwords['stopword'].tolist()
        except:
            self.stopwords = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희', '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게']

        # 버튼 이벤트 연결
        self.btn_recommend.clicked.connect(self.recommend_logic)
        self.le_keyword.returnPressed.connect(self.recommend_logic)

    def preprocess_input(self, text):
        text = re.sub('[^가-힣]', ' ', text)
        tokens = self.okt.pos(text, stem=True)
        words = []
        for word, pos in tokens:
            if pos in ['Noun', 'Verb', 'Adjective']:
                if len(word) > 1 and word not in self.stopwords:
                    words.append(word)
        return ' '.join(words)

    def recommend_logic(self):
        user_input = self.le_keyword.text().strip()
        gender = self.cb_gender.currentText()
        age = self.cb_age.currentText()
        skin_type = self.cb_skin_type.currentText()

        # 입력값 전처리
        if not user_input:
            if gender == "성별 선택" and age == "연령대 선택" and skin_type == "피부 타입 선택":
                QMessageBox.warning(self, "경고", "피부 고민을 입력하거나 조건을 선택해주세요.")
                return
            else:
                # 키워드 없이 조건만 선택된 경우: 필터링 후 첫 번째 행 표시
                filtered_df = self.df_skin.copy()
                if gender != "성별 선택":
                    filtered_df = filtered_df[filtered_df['Gender'] == gender]
                if age != "연령대 선택":
                    filtered_df = filtered_df[filtered_df['Age'] == age]
                if skin_type != "피부 타입 선택":
                    filtered_df = filtered_df[filtered_df['Skin Type'] == skin_type]
                best_idx = filtered_df.index[0] if not filtered_df.empty else 0
                self.display_result(best_idx, is_filtered=True, gender=gender, age=age, skin_type=skin_type)
                return

        cleaned_input = self.preprocess_input(user_input)
        if not cleaned_input:
            self.lb_recommendation.setHtml("죄송합니다. 입력하신 내용에서 유효한 단어를 찾을 수 없습니다.")
            return

        # [순서 변경] 기존: 프로필 3가지 조건 모두 일치하는 범위 안에서 TF-IDF 매칭
        # [순서 변경] 변경: 전체 데이터 TF-IDF 유사도 + 프로필 일치 보너스 점수 합산
        # 이유: 특정 피부 고민(예: 검버섯)이 선택한 연령대(50대 이상) 데이터에
        #       존재하지 않을 경우, 프로필 조건만 맞고 내용은 전혀 다른 항목이
        #       매칭되는 문제가 있었음.
        #       피부 고민 유사도를 주점수로 두고, 프로필 일치를 보조 점수로 처리하면
        #       유사도가 높은 항목이 프로필이 완전히 맞지 않아도 우선 선택되고,
        #       유사도가 비슷한 경우에는 프로필이 잘 맞는 항목이 선택됨.

        # 1단계: 전체 데이터에 대해 TF-IDF 코사인 유사도 계산
        input_vec = self.tfidf.transform([cleaned_input])
        cosine_sim = linear_kernel(input_vec, self.tfidf_matrix)[0]

        # 2단계: 프로필 일치 항목에 보너스 점수 부여 (항목당 0.05, 최대 0.15)
        #        보너스가 너무 크면 유사도를 역전시키므로 작은 값으로 설정
        profile_bonus = np.zeros(len(self.df_skin))
        if gender != "성별 선택":
            profile_bonus += (self.df_skin['Gender'] == gender).values * 0.05
        if age != "연령대 선택":
            profile_bonus += (self.df_skin['Age'] == age).values * 0.05
        if skin_type != "피부 타입 선택":
            profile_bonus += (self.df_skin['Skin Type'] == skin_type).values * 0.05

        # 3단계: 최종 점수 = TF-IDF 유사도 + 프로필 보너스 → 최고 점수 항목 선택
        final_score = cosine_sim + profile_bonus
        best_idx = int(final_score.argmax())

        self.display_result(best_idx, gender=gender, age=age, skin_type=skin_type)

    def display_result(self, idx, is_filtered=False, gender="성별 선택", age="연령대 선택", skin_type="피부 타입 선택"):
        row = self.df_skin.iloc[idx]
        response = row['Makeup Response']
        rec_ingredients = row['Recommended Ingredients']
        avoid_ingredients = row['Ingredients to Avoid']

        # 사용자가 선택한 프로필을 표시 (매칭된 데이터 행의 프로필이 아님)
        profile_parts = [v for v in [gender, age, skin_type] if v not in ("성별 선택", "연령대 선택", "피부 타입 선택")]
        profile_str = ", ".join(profile_parts) if profile_parts else "전체"
        profile_info = f"<small style='color: #888;'>프로필: {profile_str}</small><br>"

        result_text = profile_info
        result_text += f"<b>[추천 솔루션]</b><br>{response}<br><br>"
        result_text += f"<b>✨ 추천 성분:</b> {rec_ingredients}<br>"
        result_text += f"<b>⚠️ 주의 성분:</b> {avoid_ingredients}"

        self.lb_recommendation.setHtml(result_text)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = SkincareRecommendationApp()
    window.show()
    sys.exit(app.exec_())
