import streamlit as st
import json, re, datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="도서 추천 시스템", layout="wide")
st.title("📚 국립중앙도서관 기반 도서 추천 시스템")
st.caption("JSON 업로드 → 책 선택 또는 키워드 입력 → 유사도 + 최근 5년 가중치로 추천")

# ---------------------------
# 1) JSON 안전 로더
# ---------------------------
def safe_load_json(file):
    """
    - UTF-8 BOM 제거
    - 여러 JSON이 붙어 있는 파일에서도 '첫 번째 JSON 객체'만 파싱
    """
    text = file.read().decode("utf-8-sig")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            decoder = json.JSONDecoder()
            obj, _ = decoder.raw_decode(text.lstrip())
            return obj
        except Exception as e:
            st.error(f"❌ JSON 파일 형식 오류: {e}")
            return None

# ---------------------------
# 2) 안전 문자열 변환 유틸
# ---------------------------
def to_text(v):
    """값을 안전하게 문자열로 변환: list/dict/None/숫자 전부 OK"""
    if v is None:
        return ""
    if isinstance(v, str):
        return v
    if isinstance(v, (int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return " ".join(to_text(x) for x in v)
    if isinstance(v, dict):
        return " ".join(to_text(x) for x in v.values())
    return str(v)

# ---------------------------
# 3) 출판년도 추출 & 최근 5년 가중치
# ---------------------------
YEAR_RE = re.compile(r"(19|20)\d{2}")

def extract_year(book):
    """
    issuedYear, issued, datePublished 등에서 4자리 연도 추출.
    예) '民國77[1988]' → 1988, '2022-11-10...' → 2022
    """
    cand = [
        to_text(book.get("issuedYear")),
        to_text(book.get("issued")),
        to_text(book.get("datePublished")),
        to_text(book.get("publicationDate")),
    ]
    for c in cand:
        m = YEAR_RE.search(c)
        if m:
            try:
                return int(m.group(0))
            except:
                pass
    return None

def recency_weight(year, now_year=None):
    """
    최근 5년 선형 가중치.
    올해=1.0, 1년 전=0.8, …, 5년 이상 지난 책=0
    """
    if year is None:
        return 0.0
    if now_year is None:
        now_year = datetime.date.today().year
    d = now_year - year
    if d < 0:
        d = 0
    if d <= 5:
        return (5 - d) / 5.0
    return 0.0

# ---------------------------
# 4) 코퍼스/타이틀/메타 생성
# ---------------------------
def build_corpus(data):
    if not isinstance(data, dict):
        return [], [], [], []
    books = data.get("@graph", [])
    if not isinstance(books, list) or not books:
        return [], [], [], []

    corpus, titles, years, raw = [], [], [], []
    for bk in books:
        parts = [
            to_text(bk.get("title")),
            to_text(bk.get("remainderOfTitle")),
            to_text(bk.get("creator")),
            to_text(bk.get("dcterms:creator")),
            to_text(bk.get("subject")),
            to_text(bk.get("description")),
            to_text(bk.get("titleOfSeries")),
            to_text(bk.get("publisher")),
            to_text(bk.get("kdc")),
            to_text(bk.get("publicationPlace")),
        ]
        text = " ".join(p for p in parts if p)
        corpus.append(text)
        titles.append(to_text(bk.get("title")) or "(제목 없음)")
        years.append(extract_year(bk))
        raw.append(bk)
    return corpus, titles, years, raw

# ---------------------------
# 앱 본문
# ---------------------------
uploaded = st.file_uploader("도서정보 JSON 파일 업로드", type=["json"])
if not uploaded:
    st.info("📂 먼저 JSON 파일을 업로드하세요.")
    st.stop()

data = safe_load_json(uploaded)
if data is None:
    st.stop()

corpus, titles, years, raw_books = build_corpus(data)
if not corpus:
    st.warning("⚠️ '@graph' 내 도서 데이터를 찾지 못했습니다.")
    st.stop()

# TF-IDF 전역 학습 (책 선택형 & 키워드형에 공통 사용)
vectorizer = TfidfVectorizer()
tfidf = vectorizer.fit_transform(corpus)

# 사이드바: 가중치 설정
st.sidebar.header("⚖️ 가중치 설정")
w_recency = st.sidebar.slider("최근 5년 가중치 비율", min_value=0.0, max_value=0.8, value=0.3, step=0.05,
                              help="최종 점수 = (1-비율)*내용유사도 + (비율)*최근성")
top_n = st.sidebar.slider("추천 개수 (Top N)", 3, 15, 5)

now_year = datetime.date.today().year
recency_vec = [recency_weight(y, now_year) for y in years]

def rerank_with_recency(sim_scores):
    # sim_scores: numpy array
    # recency_vec: 파이썬 리스트 → 같은 길이 보장
    import numpy as np
    rec = np.array(recency_vec, dtype=float)
    return (1 - w_recency) * sim_scores + (w_recency) * rec

col1, col2 = st.columns(2, vertical_alignment="top")

# ---------------------------
# A) 책 선택형 추천
# ---------------------------
with col1:
    st.subheader("🔎 책 선택형 추천")
    sel_title = st.selectbox("추천 기준이 될 책을 선택하세요", options=titles)
    if st.button("이 책과 비슷한 도서 추천", use_container_width=True):
        try:
            idx = titles.index(sel_title)
        except ValueError:
            st.error("선택한 책을 찾을 수 없습니다.")
            st.stop()

        sims = cosine_similarity(tfidf[idx], tfidf).flatten()
        final = rerank_with_recency(sims)

        order = final.argsort()[::-1]
        recs = [i for i in order if i != idx][:top_n]

        st.write(f"**기준 도서:** {sel_title}")
        for i in recs:
            creator = to_text(raw_books[i].get("creator")) or "저자 정보 없음"
            kdc = to_text(raw_books[i].get("kdc")) or "N/A"
            y = years[i] or "N/A"
            st.markdown(
                f"- **{titles[i]}** — {creator} (KDC: {kdc}, 연도: {y})  "
                f"· 내용유사도: {cosine_similarity(tfidf[idx], tfidf[i]).flatten()[0]:.3f}  "
            )

# ---------------------------
# B) 키워드 검색형 추천
# ---------------------------
with col2:
    st.subheader("📝 키워드 검색형 추천")
    q = st.text_input("관심 키워드/문장을 입력하세요 (예: 도서관학, 저작권법, 역사)")
    if st.button("키워드로 추천", use_container_width=True):
        if not q.strip():
            st.warning("키워드를 입력해 주세요.")
        else:
            q_vec = vectorizer.transform([q.strip()])
            sims = cosine_similarity(q_vec, tfidf).flatten()
            final = rerank_with_recency(sims)

            order = final.argsort()[::-1][:top_n]
            st.write(f"**입력 키워드:** {q}")
            for i in order:
                creator = to_text(raw_books[i].get("creator")) or "저자 정보 없음"
                kdc = to_text(raw_books[i].get("kdc")) or "N/A"
                y = years[i] or "N/A"
                st.markdown(
                    f"- **{titles[i]}** — {creator} (KDC: {kdc}, 연도: {y})  "
                    f"· 내용유사도: {cosine_similarity(q_vec, tfidf[i]).flatten()[0]:.3f}"
                )
