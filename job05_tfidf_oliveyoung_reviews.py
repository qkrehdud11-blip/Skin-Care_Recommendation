import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.io import mmwrite
import pickle
import os

def generate_review_tfidf():
    input_file = './datasets/oliveyoung_reviews_preprocessed.csv'
    if not os.path.exists(input_file):
        print("Preprocessed file not found.")
        return

    df = pd.read_csv(input_file)
    
    # 1. TF-IDF 변환
    print("Generating TF-IDF matrix for reviews...")
    tfidf = TfidfVectorizer(sublinear_tf=True)
    tfidf_matrix = tfidf.fit_transform(df['cleaned_review'].fillna(''))
    
    # 2. 모델 및 행렬 저장
    os.makedirs("./models", exist_ok=True)
    with open('./models/tfidf_reviews.pkl', 'wb') as f:
        pickle.dump(tfidf, f)
    mmwrite('./models/Tfidf_reviews.mtx', tfidf_matrix)
    
    print(f"Saved: tfidf_reviews.pkl, Tfidf_reviews.mtx (Shape: {tfidf_matrix.shape})")

if __name__ == "__main__":
    generate_review_tfidf()
