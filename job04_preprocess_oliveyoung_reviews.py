import pandas as pd
from konlpy.tag import Okt
import re
import os

def preprocess_reviews():
    input_file = './datasets/oliveyoung_reviews.csv'
    output_file = './datasets/oliveyoung_reviews_preprocessed.csv'
    
    if not os.path.exists(input_file):
        print("Review file not found.")
        return

    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} reviews.")

    # 1. 제품별로 리뷰 묶기
    df_merged = df.groupby('product_name')['review'].apply(lambda x: ' '.join(x)).reset_index()
    print(f"Merged into {len(df_merged)} unique products.")

    # 2. 형태소 분석 및 전처리
    okt = Okt()
    
    # 불용어 로드
    try:
        df_stopwords = pd.read_csv('../movie_review/datasets/stopwords.csv')
        stopwords = df_stopwords['stopword'].tolist()
    except:
        stopwords = ['가다', '하다', '있다', '없다', '좋다', '너무', '정말']

    def clean_text(text):
        text = re.sub('[^가-힣]', ' ', text)
        tokens = okt.pos(text, stem=True)
        words = []
        for word, pos in tokens:
            if pos in ['Noun', 'Verb', 'Adjective'] and len(word) > 1:
                if word not in stopwords:
                    words.append(word)
        return ' '.join(words)

    print("Preprocessing merged reviews (this may take time with large data)...")
    df_merged['cleaned_review'] = df_merged['review'].apply(clean_text)
    
    # 3. 저장
    df_merged.to_csv(output_file, index=False)
    print(f"Saved preprocessed data to {output_file}")

if __name__ == "__main__":
    preprocess_reviews()
