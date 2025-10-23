import streamlit as st
import json, re, datetime
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="도서 추천 시스템", layout="wide")
st.title("📚 국립중앙도서관 기반 도서 추천 시스템")
st.caption("JSON 업로드 → 페이지수/연도/KDC/키워드 필터 → 책 선택형 또는 키워드형 추천 (최근 5년 가중치 반영)")

# ---------------------------
# 1) JSON 안전 로더
# ---------------------------
def safe_load_json(file):
    """UTF-8 BOM 제거 + 여러 JSON이 붙은 경우 첫 객체만 파싱"""
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
# 2) 유틸: 안전 문자열/리스트 변환
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

def to_list(v):
    """subject 같은 필드를 리스트[str]로 변환"""
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, str):
                out.append(x.strip())
            elif isinstance(x, dict):
                out.extend([to_text(val).strip() for val in x.values() if to_text(val).strip()])
            else:
                out.append(to_text(x).strip())
        return [s for s in out if s]
    if isinstance(v, dict):
        return [to_text(x).strip() for x in v.values() if to_text(x).strip()]
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    return [to_text(v).strip()]

# ---------------------------
# 3) 메타 추출: 출판년도/페이지수
# ---------------------------
YEAR_RE = re.compile(r"(19|20)\d{2}")
PAGES_RE = re.compile(r"(\d+)\s*p\b", re.IGNORECASE)  # 예: '586p'

def extract_year(book):
    """issuedYear/issued/datePublished 등에서 4자리 연도 추출"""
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

def extract_pages(book):
    """
    extent 배열 안에서 '숫자+p' 패턴 탐지 → 최대값을 페이지 수로 사용.
    예: ['21cm','x, 586p'] → 586
    """
    ext = to_list(book.get("extent"))
    pages_found = []
    for token in ext:
        for m in PAGES_RE.finditer(token):
            try:
                pages_found.append(int(m.group(1)))
            except:
                pass
    if pages_found:
        return max(pages_found)
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
def build_records(data):
    """
    각 도서 레코드를 다음 구조로 변환:
    {
      'title': str,
      'text': str (추천용 문서),
      'year': int|None,
      'pages': int|None,
      'subjects': List[str],
      'raw': dict (원본)
    }
    """
    if not isinstance(data, dict):
        return []
    books = data.get("@graph", [])
    if not isinstance(books, list) or not books:
        return []

    records = []
    for bk in books:
        title = to_text(bk.get("title")) or "(제목 없음)"
        subjects = to_list(bk.get("subject"))
        parts = [
            title,
            to_text(bk.get("remainderOfTitle")),
            to_text(bk.get("creator")),
            to_text(bk.get("dcterms:creator")),
            " ".join(subjects),
            to_text(bk.get("description")),
            to_text(bk.get("titleOfSeries")),
            to_text(bk.get("publisher")),
            to_text(bk.get("kdc")),
            to_text(bk.get("publicationPlace")),
        ]
        text = " ".join(p for p in parts if p)
        rec = {
            "title": title,
            "text": text,
            "year": extract_year(bk),
            "pages": extract_pages(bk),
            "subjects": subjects,
            "raw": bk,
        }
        records.append(rec)
    return records

# ---------------------------
# 업로드 & 변환
# ---------------------------
uploaded = st.file_uploader("도서정보 JSON 파일 업로드", type=["json"])
if not uploaded:
    st.info("📂 먼저 JSON 파일을 업로드하세요.")
    st.stop()

data = safe_load_json(uploaded)
if data is None:
    st.stop()

records = build_records(data)
if not records:
    st.warning("⚠️ '@graph' 내 도서 데이터를 찾지 못했습니다.")
    st.stop()

# ---------------------------
# 사이드바 필터: 페이지 수 + 최근 5년 가중치
# ---------------------------
pages_list = [r["pages"] for r in records if r["pages"] is not None]
if pages_list:
    min_pages, max_pages = int(min(pages_list)), int(max(pages_list))
else:
    # 페이지 정보가 전혀 없을 때 안전 기본값
    min_pages, max_pages = 0, 2000

st.sidebar.header("🔎 필터 & 가중치")
include_no_pages = st.sidebar.checkbox("쪽수 정보 없는 자료도 포함", value=True)
page_range = st.sidebar.slider("페이지(쪽) 범위", min_value=min_pages, max_value=max_pages,
                               value=(min_pages, max_pages))

w_recency = st.sidebar.slider("최근 5년 가중치 비율", 0.0, 0.8, 0.3, 0.05,
                              help="최종 점수 = (1-비율)*내용유사도 + (비율)*최근성")
top_n = st.sidebar.slider("추천 개수 (Top N)", 3, 15, 5)

# 페이지 필터 적용
def pass_page_filter(p):
    if p is None:
        return include_no_pages
    return page_range[0] <= p <= page_range[1]

filtered = [r for r in records if pass_page_filter(r["pages"])]

if not filtered:
    st.warning("⚠️ 페이지 필터 조건에 맞는 도서가 없습니다. 범위를 넓혀주세요.")
    st.stop()

# ---------------------------
# 상위 10개 키워드(주제어) 제시
# ---------------------------
all_subjects = []
for r in filtered:
    all_subjects.extend(r["subjects"])
freq = Counter([s for s in all_subjects if s])
top_keywords = [kw for kw, _ in freq.most_common(10)]

# ---------------------------
# TF-IDF (필터된 데이터 기준)
# ---------------------------
titles = [r["title"] for r in filtered]
corpus = [r["text"] for r in filtered]
years = [r["year"] for r in filtered]
pages = [r["pages"] for r in filtered]
raw_books = [r["raw"] for r in filtered]

vectorizer = TfidfVectorizer()
try:
    tfidf = vectorizer.fit_transform(corpus)
except ValueError:
    st.error("텍스트 코퍼스가 비어 있어 벡터화를 수행할 수 없습니다.")
    st.stop()

now_year = datetime.date.today().year
recency_vec = np.array([recency_weight(y, now_year) for y in years], dtype=float)

def rerank_with_recency(sim_scores):
    return (1 - w_recency) * sim_scores + (w_recency) * recency_vec

# ---------------------------
# 레이아웃: 좌(책 선택형) / 우(키워드형)
# ---------------------------
col1, col2 = st.columns(2, vertical_alignment="top")

# ----- A) 책 선택형 -----
with col1:
    st.subheader("🔖 책 선택형 추천")
    sel_title = st.selectbox("추천 기준이 될 책을 선택하세요", options=titles, index=0)
    if st.button("이 책과 비슷한 도서 추천", use_container_width=True):
        idx = titles.index(sel_title)
        sims = cosine_similarity(tfidf[idx], tfidf).flatten()
        final = rerank_with_recency(sims)
        order = final.argsort()[::-1]
        recs = [i for i in order if i != idx][:top_n]

        st.write(f"**기준 도서:** {sel_title}")
        if not recs:
            st.info("추천 결과가 없습니다.")
        else:
            for i in recs:
                creator = to_text(raw_books[i].get("creator")) or "저자 정보 없음"
                kdc = to_text(raw_books[i].get("kdc")) or "N/A"
                y = years[i] or "N/A"
                p = pages[i] if pages[i] is not None else "N/A"
                base_sim = cosine_similarity(tfidf[idx], tfidf[i]).flatten()[0]
                st.markdown(
                    f"- **{titles[i]}** — {creator} "
                    f"(KDC: {kdc}, 연도: {y}, 쪽수: {p})  · 내용유사도: {base_sim:.3f}"
                )

# ----- B) 키워드 검색형 -----
with col2:
    st.subheader("📝 키워드 검색형 추천")
    st.caption("아래에서 상위 10개 키워드를 고르거나, 직접 키워드를 입력하세요.")
    picked = st.multiselect("상위 키워드", options=top_keywords, default=[])
    q = st.text_input("자유 키워드 입력 (선택)", placeholder="예: 도서관학, 저작권법, 역사")

    if st.button("키워드로 추천", use_container_width=True):
        # 선택 키워드 + 자유 키워드 결합
        query_terms = []
        if picked:
            query_terms.append(" ".join(picked))
        if q.strip():
            query_terms.append(q.strip())
        if not query_terms:
            st.warning("키워드를 선택하거나 입력해 주세요.")
        else:
            query = " ".join(query_terms)
            q_vec = vectorizer.transform([query])
            sims = cosine_similarity(q_vec, tfidf).flatten()
            final = rerank_with_recency(sims)
            order = final.argsort()[::-1][:top_n]

            st.write(f"**입력/선택 키워드:** {query}")
            for i in order:
                creator = to_text(raw_books[i].get("creator")) or "저자 정보 없음"
                kdc = to_text(raw_books[i].get("kdc")) or "N/A"
                y = years[i] or "N/A"
                p = pages[i] if pages[i] is not None else "N/A"
                base_sim = cosine_similarity(q_vec, tfidf[i]).flatten()[0]
                st.markdown(
                    f"- **{titles[i]}** — {creator} "
                    f"(KDC: {kdc}, 연도: {y}, 쪽수: {p})  · 내용유사도: {base_sim:.3f}"
                )
