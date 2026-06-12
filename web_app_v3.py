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
    DATABASE=Path(app.instance_path) / 'users_v3.db',
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE='Lax',
    PERMANENT_SESSION_LIFETIME=60 * 60 * 8,
)
Path(app.instance_path).mkdir(parents=True, exist_ok=True)

print('Loading data and models...')
df_skin = pd.read_csv(BASE_DIR / 'datasets' / 'skin_data_final.csv')
tfidf_matrix = mmread(BASE_DIR / 'models' / 'Tfidf_skin_data.mtx').tocsr()
with open(BASE_DIR / 'models' / 'tfidf.pkl', 'rb') as model_file:
    tfidf = pickle.load(model_file)
okt = Okt()

try:
    df_stopwords = pd.read_csv(BASE_DIR.parent / 'movie_review' / 'datasets' / 'stopwords.csv')
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
         str(row['Recommended Ingredients']), utc_now()),
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
        'index_3.html', recent_history=recent_history,
        popular_terms=get_popular_terms(),
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
        gender = request.form.get('gender', '')
        age = request.form.get('age', '')
        skin_type = request.form.get('skin_type', '')
        error = validate_registration(username, name, email, password, password_confirm)
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
    return render_template('auth_3.html', mode='register')


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
    return render_template('auth_3.html', mode='login')


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
        'admin_3.html', users=users, stats=stats,
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
    gender = data.get('gender', '성별 선택')
    age = data.get('age', '연령대 선택')
    skin_type = data.get('skin_type', '피부 타입 선택')

    cleaned = ''
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
    else:
        cleaned = preprocess_input(user_input)
        if not cleaned:
            return jsonify({'error': '입력하신 내용에서 유효한 단어를 찾을 수 없습니다.'}), 400
        input_vec = tfidf.transform([cleaned])
        cosine_sim = linear_kernel(input_vec, tfidf_matrix)[0]
        profile_bonus = np.zeros(len(df_skin))
        if gender != '성별 선택':
            profile_bonus += (df_skin['Gender'] == gender).values * 0.05
        if age != '연령대 선택':
            profile_bonus += (df_skin['Age'] == age).values * 0.05
        if skin_type != '피부 타입 선택':
            profile_bonus += (df_skin['Skin Type'] == skin_type).values * 0.05
        row = df_skin.iloc[int((cosine_sim + profile_bonus).argmax())]

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
    return build_response(row, gender, age, skin_type)


def build_response(row, gender, age, skin_type):
    profile_parts = [value for value in [gender, age, skin_type]
                     if value not in ('성별 선택', '연령대 선택', '피부 타입 선택')]
    return jsonify({
        'profile': ', '.join(profile_parts) if profile_parts else '전체',
        'response': row['Makeup Response'],
        'rec_ingredients': row['Recommended Ingredients'],
        'avoid_ingredients': row['Ingredients to Avoid'],
    })


with app.app_context():
    init_db()


if __name__ == '__main__':
    app.run(debug=True, port=5000)
