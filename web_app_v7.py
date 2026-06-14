import os
import pickle
import re
import secrets
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from flask import (
    Flask, abort, flash, g, jsonify, redirect, render_template, request,
    session, url_for,
)
from konlpy.tag import Okt
from scipy.io import mmread
from sklearn.metrics.pairwise import linear_kernel
from werkzeug.security import check_password_hash, generate_password_hash

BASE_DIR = Path(__file__).resolve().parent


def load_secret_key(instance_path):
    env_secret = os.environ.get('SKINSCOPE_SECRET_KEY')
    if env_secret:
        return env_secret
    secret_file = Path(instance_path) / 'secret_key'
    secret_file.parent.mkdir(parents=True, exist_ok=True)
    if not secret_file.exists():
        secret_file.write_text(secrets.token_hex(32), encoding='ascii')
    return secret_file.read_text(encoding='ascii').strip()


app = Flask(__name__, instance_relative_config=True)
app.config.update(
    SECRET_KEY=load_secret_key(app.instance_path),
    DATABASE=Path(app.instance_path) / 'users_v7.db',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
)
Path(app.instance_path).mkdir(parents=True, exist_ok=True)

print('Loading data and models...')
df_skin = pd.read_csv(BASE_DIR / 'datasets' / 'skin_data_final.csv')
tfidf_matrix = mmread(BASE_DIR / 'models' / 'Tfidf_skin_data_rebuild.mtx').tocsr()
with open(BASE_DIR / 'models' / 'tfidf_rebuild.pkl', 'rb') as model_file:
    tfidf = pickle.load(model_file)
required_skin_columns = {
    'Gender', 'Age', 'Skin Type', 'User Question', 'Makeup Response',
    'Recommended Ingredients', 'Ingredients to Avoid', 'cleaned_question',
}
missing_skin_columns = required_skin_columns.difference(df_skin.columns)
if missing_skin_columns:
    raise RuntimeError(f'피부 상담 데이터 필수 열이 없습니다: {sorted(missing_skin_columns)}')
if tfidf_matrix.shape[0] != len(df_skin):
    raise RuntimeError('피부 상담 데이터와 rebuild TF-IDF 행 수가 일치하지 않습니다.')
if tfidf_matrix.shape[1] != len(tfidf.get_feature_names_out()):
    raise RuntimeError('rebuild TF-IDF 모델과 행렬의 특성 수가 일치하지 않습니다.')

SKINCARE_CATEGORIES = {'스킨/토너', '에센스/세럼/앰플', '크림', '로션'}
PROFILE_OPTIONS = {
    'gender': {'성별 선택', '남성', '여성'},
    'age': {'연령대 선택', '20대', '30대', '40대', '50대 이상'},
    'skin_type': {'피부 타입 선택', '건성', '정상', '지성', '복합성'},
}
PROFILE_LABELS = {
    'gender': '성별',
    'age': '연령대',
    'skin_type': '피부 타입',
}
CONCERN_GROUPS = (
    ({'건조', '당김', '당기', '푸석', '각질'}, ('건조', '당기', '푸석', '각질')),
    ({'민감', '붉', '홍조', '주사'}, ('민감', '붉', '홍조', '주사')),
    ({'트러블', '여드름', '면포', '화농'}, ('트러블', '여드름', '면포', '화농')),
    ({'모공', '유분', '피지', '블랙헤드', '번들'}, ('모공', '유분', '피지', '블랙헤드', '번들')),
    ({'주름', '탄력', '노화'}, ('주름', '탄력', '노화')),
    ({'기미', '주근깨', '색소', '착색'}, ('기미', '주근깨', '색소', '착색')),
)
QUERY_NOISE_TERMS = {'피부', '고민', '생기다', '자주'}
df_product_list = None
df_reviews = None
tfidf_matrix_reviews = None
tfidf_reviews = None
try:
    product_data = pd.read_csv(BASE_DIR / 'datasets' / 'oliveyoung_product_list.csv')
    review_data = pd.read_csv(BASE_DIR / 'datasets' / 'oliveyoung_reviews_preprocessed.csv')
    review_matrix = mmread(BASE_DIR / 'models' / 'Tfidf_reviews.mtx').tocsr()
    with open(BASE_DIR / 'models' / 'tfidf_reviews.pkl', 'rb') as model_file:
        review_vectorizer = pickle.load(model_file)

    product_data = product_data[product_data['category'].isin(SKINCARE_CATEGORIES)].copy()
    product_lookup = product_data[
        ['product_name', 'product_brand', 'product_link', 'category']
    ].drop_duplicates(subset='product_name', keep='first')
    review_data = review_data.merge(product_lookup, on='product_name', how='left')
    skincare_mask = review_data['category'].isin(SKINCARE_CATEGORIES).to_numpy()
    if review_matrix.shape[0] != len(review_data):
        raise ValueError('리뷰 CSV와 TF-IDF 행 수가 일치하지 않습니다.')
    if review_matrix.shape[1] != len(review_vectorizer.get_feature_names_out()):
        raise ValueError('리뷰 TF-IDF 모델과 행렬의 특성 수가 일치하지 않습니다.')

    df_product_list = product_data.reset_index(drop=True)
    df_reviews = review_data.loc[skincare_mask].reset_index(drop=True)
    tfidf_matrix_reviews = review_matrix[skincare_mask]
    tfidf_reviews = review_vectorizer
except (FileNotFoundError, KeyError, ValueError, pickle.UnpicklingError) as exc:
    print(f'Review recommendation unavailable: {exc}')

okt = Okt()

try:
    df_stopwords = pd.read_csv(BASE_DIR / 'datasets' / 'stopwords.csv')
    stopwords = df_stopwords['stopword'].tolist()
except (FileNotFoundError, KeyError):
    stopwords = ['아', '휴', '아이구', '아이쿠', '아이고', '어', '나', '우리', '저희',
                 '따라', '의해', '을', '를', '에', '의', '가', '으로', '로', '에게']
print('Load complete.')


def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(app.config['DATABASE'])
        g.db.row_factory = sqlite3.Row
        g.db.execute('PRAGMA foreign_keys = ON')
    return g.db


@app.teardown_appcontext
def close_db(_error=None):
    db = g.pop('db', None)
    if db is not None:
        db.close()


def init_db():
    db = get_db()
    db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE COLLATE NOCASE,
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL DEFAULT 'user' CHECK(role IN ('user', 'admin')),
            gender TEXT,
            age TEXT,
            skin_type TEXT,
            recommendation_count INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            last_login TEXT
        );
        CREATE TABLE IF NOT EXISTS search_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            keyword TEXT,
            search_terms TEXT,
            gender TEXT,
            age TEXT,
            skin_type TEXT,
            recommended_ingredients TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        );
        CREATE INDEX IF NOT EXISTS idx_search_history_user_id ON search_history(user_id);
        CREATE INDEX IF NOT EXISTS idx_search_history_created_at ON search_history(created_at DESC);
    ''')
    admin_username = os.environ.get('SKINSCOPE_ADMIN_USERNAME', 'admin').strip().lower()
    admin_password = os.environ.get('SKINSCOPE_ADMIN_PASSWORD', 'Admin1234!')
    admin_email = os.environ.get('SKINSCOPE_ADMIN_EMAIL', 'admin@skinscope.local').strip().lower()
    existing = db.execute('SELECT id FROM users WHERE username = ?', (admin_username,)).fetchone()
    if existing is None:
        db.execute(
            '''INSERT INTO users
               (username, name, email, password_hash, role, created_at)
               VALUES (?, ?, ?, ?, 'admin', ?)''',
            (admin_username, '관리자', admin_email, generate_password_hash(admin_password), utc_now()),
        )
        db.commit()
        print(f"Admin account created: {admin_username}")


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def csrf_token():
    if '_csrf_token' not in session:
        session['_csrf_token'] = secrets.token_urlsafe(32)
    return session['_csrf_token']


app.jinja_env.globals['csrf_token'] = csrf_token


@app.template_filter('kst')
def format_kst(value, date_format='%Y-%m-%d %H:%M'):
    if not value:
        return '-'
    parsed = datetime.fromisoformat(value)
    return parsed.astimezone(ZoneInfo('Asia/Seoul')).strftime(date_format)


def validate_csrf():
    submitted = request.form.get('csrf_token', '')
    if not submitted or not secrets.compare_digest(submitted, session.get('_csrf_token', '')):
        abort(400, description='유효하지 않은 요청입니다. 페이지를 새로고침해 주세요.')


def is_safe_redirect(target):
    host_url = urlparse(request.host_url)
    redirect_url = urlparse(urljoin(request.host_url, target))
    return redirect_url.scheme in ('http', 'https') and host_url.netloc == redirect_url.netloc


def login_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash('로그인이 필요한 서비스입니다.', 'info')
            return redirect(url_for('login', next=request.path))
        return view(**kwargs)
    return wrapped_view


def admin_required(view):
    @wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            flash('관리자 로그인이 필요합니다.', 'info')
            return redirect(url_for('login', next=request.path))
        if g.user['role'] != 'admin':
            abort(403)
        return view(**kwargs)
    return wrapped_view


@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')
    g.user = None if user_id is None else get_db().execute(
        '''SELECT id, username, name, email, role, gender, age, skin_type,
                  recommendation_count, created_at, last_login
           FROM users WHERE id = ?''',
        (user_id,),
    ).fetchone()


@app.context_processor
def inject_user():
    return {'current_user': g.user}


def preprocess_input(text):
    text = re.sub('[^가-힣]', ' ', text)
    tokens = okt.pos(text, stem=True)
    words = [word for word, part in tokens
             if part in ['Noun', 'Verb', 'Adjective'] and len(word) > 1 and word not in stopwords]
    return ' '.join(words)


def search_text(cleaned):
    focused = ' '.join(
        term for term in cleaned.split() if term not in QUERY_NOISE_TERMS
    )
    return focused or cleaned


def select_expert_row(user_input, cleaned, gender, age, skin_type):
    similarities = linear_kernel(
        tfidf.transform([search_text(cleaned)]), tfidf_matrix,
    )[0]
    if float(similarities.max()) <= 0:
        return None

    candidate_mask = np.ones(len(df_skin), dtype=bool)
    for column, value, default in (
        ('Gender', gender, '성별 선택'),
        ('Age', age, '연령대 선택'),
        ('Skin Type', skin_type, '피부 타입 선택'),
    ):
        if value != default:
            candidate_mask &= df_skin[column].eq(value).to_numpy()
    if not candidate_mask.any():
        candidate_mask[:] = True

    concern_bonus = np.zeros(len(df_skin))
    for triggers, dataset_terms in CONCERN_GROUPS:
        if any(trigger in user_input for trigger in triggers):
            for term in dataset_terms:
                concern_bonus += (
                    df_skin['User Question'].str.contains(term, na=False).to_numpy() * 0.04
                )

    candidate_indices = np.flatnonzero(candidate_mask)
    scores = similarities[candidate_indices] + concern_bonus[candidate_indices]
    return df_skin.iloc[int(candidate_indices[int(scores.argmax())])]


def validated_profile_value(field, value):
    value = str(value)
    defaults = {
        'gender': '성별 선택',
        'age': '연령대 선택',
        'skin_type': '피부 타입 선택',
    }
    return value if value in PROFILE_OPTIONS[field] else defaults[field]


def validate_optional_profile(field, value):
    value = str(value).strip()
    if not value:
        return None, None
    if value not in PROFILE_OPTIONS[field] or value.endswith('선택'):
        return None, f"올바른 {PROFILE_LABELS[field]} 값을 선택해 주세요."
    return value, None


def normalize_popular_term(term):
    if term.endswith('하다') and len(term) > 2:
        return term[:-2]
    return term


def get_popular_terms(limit=5):
    rows = get_db().execute(
        '''SELECT search_terms FROM search_history
           WHERE search_terms IS NOT NULL AND search_terms != ''
           ORDER BY id DESC LIMIT 1000'''
    ).fetchall()
    counter = Counter()
    ignored_terms = {'피부', '고민', '하다', '되다', '있다', '생기다', '자주'}
    for row in rows:
        terms = {normalize_popular_term(term) for term in row['search_terms'].split()
                 if term not in ignored_terms}
        counter.update(terms)
    return counter.most_common(limit)


def record_search(user_input, cleaned, gender, age, skin_type, row):
    db = get_db()
    db.execute(
        '''INSERT INTO search_history
           (user_id, keyword, search_terms, gender, age, skin_type,
            recommended_ingredients, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
        (g.user['id'] if g.user is not None else None,
         user_input or None, cleaned or None,
         None if gender == '성별 선택' else gender,
         None if age == '연령대 선택' else age,
         None if skin_type == '피부 타입 선택' else skin_type,
         str(row['Recommended Ingredients']) if row is not None else '리뷰 기반 제품 추천',
         utc_now()),
    )
    db.commit()


@app.route('/')
def index():
    recent_history = []
    if g.user is not None:
        recent_history = get_db().execute(
            '''SELECT keyword, gender, age, skin_type, recommended_ingredients, created_at
               FROM search_history WHERE user_id = ?
               ORDER BY id DESC LIMIT 5''',
            (g.user['id'],),
        ).fetchall()
    return render_template(
        'index_7.html', recent_history=recent_history,
        popular_terms=get_popular_terms(),
    )


@app.route('/analysis')
def analysis():
    return render_template(
        'analysis_7.html',
        review_mode_available=df_reviews is not None and len(df_reviews) > 0,
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    if g.user is not None:
        return redirect(url_for('index'))
    if request.method == 'POST':
        validate_csrf()
        username = request.form.get('username', '').strip().lower()
        name = request.form.get('name', '').strip()
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        password_confirm = request.form.get('password_confirm', '')
        gender, gender_error = validate_optional_profile('gender', request.form.get('gender', ''))
        age, age_error = validate_optional_profile('age', request.form.get('age', ''))
        skin_type, skin_type_error = validate_optional_profile(
            'skin_type', request.form.get('skin_type', ''),
        )
        error = validate_registration(username, name, email, password, password_confirm)
        error = error or gender_error or age_error or skin_type_error
        if error is None:
            try:
                db = get_db()
                db.execute(
                    '''INSERT INTO users
                       (username, name, email, password_hash, role, gender, age, skin_type, created_at)
                       VALUES (?, ?, ?, ?, 'user', ?, ?, ?, ?)''',
                    (username, name, email, generate_password_hash(password),
                     gender or None, age or None, skin_type or None, utc_now()),
                )
                db.commit()
            except sqlite3.IntegrityError:
                error = '이미 사용 중인 아이디 또는 이메일입니다.'
            else:
                flash('회원가입이 완료되었습니다. 로그인해 주세요.', 'success')
                return redirect(url_for('login'))
        flash(error, 'error')
    return render_template('auth_7.html', mode='register')


def validate_registration(username, name, email, password, password_confirm):
    if not re.fullmatch(r'[a-z0-9_]{4,20}', username):
        return '아이디는 영문 소문자, 숫자, 밑줄을 사용해 4~20자로 입력해 주세요.'
    if not 2 <= len(name) <= 30:
        return '이름은 2~30자로 입력해 주세요.'
    if not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', email):
        return '올바른 이메일 주소를 입력해 주세요.'
    if len(password) < 8 or not re.search(r'[A-Za-z]', password) or not re.search(r'\d', password):
        return '비밀번호는 영문과 숫자를 포함해 8자 이상이어야 합니다.'
    if password != password_confirm:
        return '비밀번호 확인이 일치하지 않습니다.'
    return None


@app.route('/login', methods=['GET', 'POST'])
def login():
    if g.user is not None:
        return redirect(url_for('index'))
    if request.method == 'POST':
        validate_csrf()
        username = request.form.get('username', '').strip().lower()
        password = request.form.get('password', '')
        user = get_db().execute('SELECT * FROM users WHERE username = ?', (username,)).fetchone()
        if user is None or not check_password_hash(user['password_hash'], password):
            flash('아이디 또는 비밀번호가 올바르지 않습니다.', 'error')
        else:
            session.clear()
            session.permanent = True
            session['user_id'] = user['id']
            get_db().execute('UPDATE users SET last_login = ? WHERE id = ?', (utc_now(), user['id']))
            get_db().commit()
            next_url = request.args.get('next', '')
            if next_url and is_safe_redirect(next_url):
                return redirect(next_url)
            return redirect(url_for('admin_users' if user['role'] == 'admin' else 'index'))
    return render_template('auth_7.html', mode='login')


@app.post('/logout')
@login_required
def logout():
    validate_csrf()
    session.clear()
    flash('로그아웃되었습니다.', 'success')
    return redirect(url_for('index'))


@app.route('/admin/users')
@admin_required
def admin_users():
    users = get_db().execute(
        '''SELECT id, username, name, email, role, gender, age, skin_type,
                  recommendation_count, created_at, last_login
           FROM users ORDER BY id DESC'''
    ).fetchall()
    stats = get_db().execute(
        '''SELECT COUNT(*) AS total,
                  SUM(CASE WHEN role = 'user' THEN 1 ELSE 0 END) AS members,
                  COALESCE(SUM(recommendation_count), 0) AS recommendations,
                  SUM(CASE WHEN last_login IS NOT NULL THEN 1 ELSE 0 END) AS active,
                  (SELECT COUNT(*) FROM search_history) AS searches
           FROM users'''
    ).fetchone()
    recent_searches = get_db().execute(
        '''SELECT search_history.id, search_history.keyword, search_history.gender,
                  search_history.age, search_history.skin_type,
                  search_history.recommended_ingredients, search_history.created_at,
                  users.name, users.username
           FROM search_history
           LEFT JOIN users ON users.id = search_history.user_id
           ORDER BY search_history.id DESC LIMIT 20'''
    ).fetchall()
    return render_template(
        'admin_7.html', users=users, stats=stats,
        recent_searches=recent_searches, popular_terms=get_popular_terms(),
    )


@app.post('/admin/users/<int:user_id>/delete')
@admin_required
def delete_user(user_id):
    validate_csrf()
    db = get_db()
    user = db.execute(
        'SELECT id, username, name, role FROM users WHERE id = ?',
        (user_id,),
    ).fetchone()
    if user is None:
        flash('삭제할 사용자를 찾을 수 없습니다.', 'error')
        return redirect(url_for('admin_users'))
    if user['id'] == g.user['id']:
        flash('현재 로그인한 관리자 계정은 삭제할 수 없습니다.', 'error')
        return redirect(url_for('admin_users'))
    if user['role'] == 'admin':
        flash('관리자 계정은 사용자 목록에서 삭제할 수 없습니다.', 'error')
        return redirect(url_for('admin_users'))
    db.execute('DELETE FROM users WHERE id = ?', (user_id,))
    db.commit()
    flash(f"{user['name']}({user['username']}) 계정을 삭제했습니다.", 'success')
    return redirect(url_for('admin_users'))


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True) or {}
    user_input = str(data.get('keyword', '')).strip()[:500]
    gender = validated_profile_value('gender', data.get('gender', '성별 선택'))
    age = validated_profile_value('age', data.get('age', '연령대 선택'))
    skin_type = validated_profile_value(
        'skin_type', data.get('skin_type', '피부 타입 선택'),
    )
    mode = str(data.get('mode', 'expert'))

    if mode == 'review':
        return recommend_by_review(user_input, gender, age, skin_type)
    if mode != 'expert':
        return jsonify({'error': '지원하지 않는 추천 방식입니다.'}), 400

    cleaned = ''
    if not user_input:
        if all(value.endswith('선택') for value in (gender, age, skin_type)):
            return jsonify({'error': '피부 고민 또는 프로필을 하나 이상 입력해 주세요.'}), 400
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
    else:
        cleaned = preprocess_input(user_input)
        if not cleaned:
            return jsonify({'error': '입력하신 내용에서 유효한 단어를 찾을 수 없습니다.'}), 400
        row = select_expert_row(user_input, cleaned, gender, age, skin_type)
        if row is None:
            return jsonify({
                'error': '입력한 고민과 연결되는 상담 데이터를 찾지 못했습니다. 다른 표현으로 입력해 주세요.',
            }), 404

    if g.user is not None:
        db = get_db()
        db.execute(
            '''UPDATE users SET gender = ?, age = ?, skin_type = ?,
                                recommendation_count = recommendation_count + 1
               WHERE id = ?''',
            (None if gender == '성별 선택' else gender,
             None if age == '연령대 선택' else age,
             None if skin_type == '피부 타입 선택' else skin_type,
             g.user['id']),
        )
        db.commit()
    record_search(user_input, cleaned, gender, age, skin_type, row)
    return build_response(row, user_input, gender, age, skin_type)


def safe_product_url(value):
    parsed = urlparse(str(value))
    if parsed.scheme == 'https' and parsed.hostname in {
        'www.oliveyoung.co.kr', 'oliveyoung.co.kr',
    }:
        return parsed.geturl()
    return None


def update_member_profile(gender, age, skin_type):
    if g.user is None:
        return
    db = get_db()
    db.execute(
        """UPDATE users SET gender = ?, age = ?, skin_type = ?,
                            recommendation_count = recommendation_count + 1
           WHERE id = ?""",
        (None if gender == '성별 선택' else gender,
         None if age == '연령대 선택' else age,
         None if skin_type == '피부 타입 선택' else skin_type,
         g.user['id']),
    )
    db.commit()


def recommend_by_review(user_input, gender, age, skin_type):
    if df_reviews is None or tfidf_reviews is None or tfidf_matrix_reviews is None:
        return jsonify({'error': '현재 리뷰 추천 데이터를 사용할 수 없습니다.'}), 503
    if not user_input:
        return jsonify({'error': '리뷰 추천에는 피부 고민 입력이 필요합니다.'}), 400

    cleaned = preprocess_input(user_input)
    if not cleaned:
        return jsonify({'error': '입력하신 내용에서 유효한 단어를 찾을 수 없습니다.'}), 400

    similarities = linear_kernel(
        tfidf_reviews.transform([search_text(cleaned)]),
        tfidf_matrix_reviews,
    )[0]
    ranked_indices = similarities.argsort()[::-1]
    results = []
    seen_products = set()
    for index in ranked_indices:
        score = float(similarities[index])
        if score <= 0:
            break
        row = df_reviews.iloc[int(index)]
        product_name = str(row['product_name'])
        if product_name in seen_products:
            continue
        product_url = safe_product_url(row['product_link'])
        if not product_url:
            continue
        snippet = re.sub(r'\s+', ' ', str(row['review'])).strip()
        results.append({
            'name': product_name,
            'brand': str(row['product_brand']),
            'category': str(row['category']),
            'link': product_url,
            'review_snippet': snippet[:180] + ('...' if len(snippet) > 180 else ''),
            'match_percent': round(min(score, 1.0) * 100),
        })
        seen_products.add(product_name)
        if len(results) == 3:
            break

    if not results:
        return jsonify({'error': '해당 고민과 연결되는 스킨케어 리뷰를 찾지 못했습니다.'}), 404
    update_member_profile(gender, age, skin_type)
    record_search(user_input, cleaned, gender, age, skin_type, None)
    return jsonify({'mode': 'review', 'results': results})


def find_related_products(user_input, recommended_ingredients):
    if df_product_list is None:
        return []
    terms = []
    for text in (user_input, str(recommended_ingredients)):
        if text and text != 'nan':
            terms.extend(okt.nouns(text))
    terms = [
        term for term in dict.fromkeys(terms)
        if len(term) >= 2 and term not in {'피부', '추출물', '사용', '도움', '성분'}
    ]

    results = []
    seen_links = set()
    for term in terms[:6]:
        matches = df_product_list[
            df_product_list['product_name'].str.contains(term, case=False, na=False)
            | df_product_list['product_brand'].str.contains(term, case=False, na=False)
        ]
        for _, product in matches.head(2).iterrows():
            product_url = safe_product_url(product['product_link'])
            if not product_url or product_url in seen_links:
                continue
            results.append({
                'brand': str(product['product_brand']),
                'name': str(product['product_name']),
                'category': str(product['category']),
                'link': product_url,
            })
            seen_links.add(product_url)
            if len(results) == 6:
                return results
    return results


def build_response(row, user_input, gender, age, skin_type):
    profile_parts = [value for value in [gender, age, skin_type]
                     if value not in ('성별 선택', '연령대 선택', '피부 타입 선택')]
    return jsonify({
        'mode': 'expert',
        'profile': ', '.join(profile_parts) if profile_parts else '전체',
        'response': row['Makeup Response'],
        'rec_ingredients': row['Recommended Ingredients'],
        'avoid_ingredients': row['Ingredients to Avoid'],
        'product_recommendations': find_related_products(
            user_input, row['Recommended Ingredients'],
        ),
    })


with app.app_context():
    init_db()


if __name__ == '__main__':
    debug_enabled = os.environ.get('SKINSCOPE_DEBUG', '').lower() in {'1', 'true', 'yes'}
    app.run(
        host=os.environ.get('SKINSCOPE_HOST', '127.0.0.1'),
        port=int(os.environ.get('SKINSCOPE_PORT', '5004')),
        debug=debug_enabled,
    )
