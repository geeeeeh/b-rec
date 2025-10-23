import streamlit as st
import json, re, datetime
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 기본 세팅
# =========================
st.set_page_config(page_title="국립중앙도서관 기반 도서 추천 시스템", layout="wide")
st.title("📚 국립중앙도서관 기반 도서 추천 시스템")
st.caption("JSON 업로드 → 페이지/키워드 필터 → 책 검색형 / 키워드형 추천 · 주제/설명/저자/출판사 가중치 + 출간일 최근 5년 가중치")

# =========================
# 안전 로더 / 유틸
# =========================
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

def to_text(v):
    """값을 안전하게 문자열로 변환"""
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
                s = x.strip()
                if s: out.append(s)
            elif isinstance(x, dict):
                for val in x.values():
                    s = to_text(val).strip()
                    if s: out.append(s)
            else:
                s = to_text(x).strip()
                if s: out.append(s)
        return out
    if isinstance(v, dict):
        return [to_text(x).strip() for x in v.values() if to_text(x).strip()]
    if isinstance(v, str):
        return [v.strip()] if v.strip() else []
    s = to_text(v).strip()
    return [s] if s else []

YEAR_RE = re.compile(r"(19|20)\d{2}")
PAGES_RE = re.compile(r"(\d+)\s*p\b", re.IGNORECASE)

def extract_year(book):
    """issuedYear/issued/datePublished 등에서 4자리 연도 추출"""
    for c in [to_text(book.get("issuedYear")),
              to_text(book.get("issued")),
              to_text(book.get("datePublished")),
              to_text(book.get("publicationDate"))]:
        m = YEAR_RE.search(c)
        if m:
            try: return int(m.group(0))
            except: pass
    return None

def extract_pages(book):
    """extent에서 '숫자+p' 추출 → 최대값을 페이지 수로 사용"""
    ext = to_list(book.get("extent"))
    found = []
    for token in ext:
        for m in PAGES_RE.finditer(token):
            try: found.append(int(m.group(1)))
            except: pass
    return max(found) if found else None

def recency_weight(year, now_year=None):
    """최근 5년 선형 가중치 (올해=1.0 … 5년 이상=0)"""
    if year is None: return 0.0
    if now_year is None: now_year = datetime.date.today().year
    d = now_year - year
    d = max(d, 0)
    if d <= 5: return (5 - d) / 5.0
    return 0.0

# 추천 결과에 붙일 관련 키워드 2~3개 선정
def pick_related_keywords(subjects, picked_keywords=None, top_n=3):
    subs = [s for s in subjects if s]
    if not subs:
        return []
    picked_set = set([k.strip() for k in (picked_keywords or []) if k.strip()])
    # 1순위: 사용자가 선택한 키워드와의 교집합
    inter = [s for s in subs if s in picked_set]
    result = inter[:top_n]
    # 부족하면 subject에서 앞쪽 키워드로 채우기
    if len(result) < top_n:
        for s in subs:
            if s not in result:
                result.append(s)
            if len(result) >= top_n:
                break
    return result[:top_n]

# =========================
# 데이터 변환
# =========================
def build_records(data):
    """
    각 도서 레코드:
    {
      'title': str,
      'subjects': List[str],
      'desc': str,
      'creator': str,
      'publisher': str,
      'year': int|None,
      'pages': int|None,
      'raw': dict
    }
    """
    if not isinstance(data, dict): return []
    books = data.get("@graph", [])
    if not isinstance(books, list) or not books: return []
    recs = []
    for bk in books:
        recs.append({
            "title": to_text(bk.get("title")) or "(제목 없음)",
            "subjects": to_list(bk.get("subject")),
            "desc": to_text(bk.get("description")),
            "creator": to_text(bk.get("creator")),
            "publisher": to_text(bk.get("publisher")),
            "year": extract_year(bk),
            "pages": extract_pages(bk),
            "raw": bk,
        })
    return recs

# =========================
# 업로드
# =========================
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

# =========================
# 페이지 필터
# =========================
pages_list = [r["pages"] for r in records if r["pages"] is not None]
min_pages, max_pages = (int(min(pages_list)), int(max(pages_list))) if pages_list else (0, 2000)

st.sidebar.header("🔎 필터 & 가중치")
include_no_pages = st.sidebar.checkbox("쪽수 정보 없는 자료도 포함", value=True)
page_range = st.sidebar.slider("페이지(쪽) 범위", min_value=min_pages, max_value=max_pages,
                               value=(min_pages, max_pages))

def pass_page_filter(p):
    if p is None:
        return include_no_pages
    return page_range[0] <= p <= page_range[1]

filtered = [r for r in records if pass_page_filter(r["pages"])]
if not filtered:
    st.warning("⚠️ 페이지 필터 조건에 맞는 도서가 없습니다. 범위를 넓혀주세요.")
    st.stop()

# =========================
# 상위 10 키워드
# =========================
all_subjects = []
for r in filtered:
    all_subjects.extend(r["subjects"])
top_keywords = [kw for kw, _ in Counter([s for s in all_subjects if s]).most_common(10)]

# =========================
# 필드별 말뭉치 (제목/KDC 제외)
# =========================
titles = [r["title"] for r in filtered]
subject_texts = [" ".join(r["subjects"]) for r in filtered]
desc_texts = [r["desc"] for r in filtered]
author_texts = [r["creator"] for r in filtered]
publisher_texts = [r["publisher"] for r in filtered]
years = [r["year"] for r in filtered]
pages = [r["pages"] for r in filtered]
raw_books = [r["raw"] for r in filtered]
subjects_by_idx = [r["subjects"] for r in filtered]

# 벡터라이저(필드별)
vec_subj = TfidfVectorizer()
vec_desc = TfidfVectorizer()
vec_auth = TfidfVectorizer()
vec_pub  = TfidfVectorizer()

X_subj  = vec_subj.fit_transform(subject_texts)
X_desc  = vec_desc.fit_transform(desc_texts)
X_auth  = vec_auth.fit_transform(author_texts)
X_pub   = vec_pub.fit_transform(publisher_texts)

now_year = datetime.date.today().year
recency_vec = np.array([recency_weight(y, now_year) for y in years], dtype=float)

# =========================
# 가중치 UI (제목/KDC 제거)
# =========================
st.sidebar.markdown("### ⚖️ 세부 가중치 (콘텐츠)")
w_subj = st.sidebar.slider("주제(키워드) 가중치", 0.0, 1.0, 0.45, 0.05)
w_desc = st.sidebar.slider("설명(요약) 가중치", 0.0, 1.0, 0.30, 0.05)
w_auth = st.sidebar.slider("저자 가중치", 0.0, 1.0, 0.15, 0.05)
w_pub  = st.sidebar.slider("출판사 가중치", 0.0, 1.0, 0.10, 0.05)

w_sum = w_subj + w_desc + w_auth + w_pub
if w_sum == 0:
    w_subj, w_desc, w_auth, w_pub = 0.45, 0.30, 0.15, 0.10
    w_sum = 1.0
w_subj, w_desc, w_auth, w_pub = [w / w_sum for w in [w_subj, w_desc, w_auth, w_pub]]

st.sidebar.markdown("### ⏱ 출간일 최근 5년 가중치")
w_recency = st.sidebar.slider("출간일 최근 5년 가중치", 0.0, 0.8, 0.30, 0.05,
                              help="최종 점수 = (1-비율)*콘텐츠점수 + (비율)*최근성")
top_n = st.sidebar.slider("추천 개수 (Top N)", 3, 15, 5)

def combine_content_score(s_subj, s_desc, s_auth, s_pub):
    return (w_subj*s_subj + w_desc*s_desc + w_auth*s_auth + w_pub*s_pub)

def final_score(content_sim, rec_vec):
    return (1 - w_recency) * content_sim + w_recency * rec_vec

# =========================
# 레이아웃
# =========================
col1, col2 = st.columns(2, vertical_alignment="top")

# ---------- A) 책 선택형 (검색 기반) ----------
with col1:
    st.subheader("🔖 책 검색형 추천 ")
    # 검색어 입력
    query_title = st.text_input("제목 검색어를 입력하세요 (부분일치 지원)", placeholder="예: 도서관학, 저작권, 디지털도서관")
    # 검색 실행 버튼
    if "matched_indices" not in st.session_state:
        st.session_state.matched_indices = []

    if st.button("검색"):
        q = query_title.strip().lower()
        if not q:
            st.warning("검색어를 입력하세요.")
            st.session_state.matched_indices = []
        else:
            matches = [i for i, t in enumerate(titles) if q in t.lower()]
            if not matches:
                st.info("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")
            st.session_state.matched_indices = matches

    # 검색 결과 목록에서 선택
    if st.session_state.matched_indices:
        options = [titles[i] for i in st.session_state.matched_indices]
        sel_title = st.selectbox("검색 결과에서 기준 도서를 선택하세요", options=options, index=0, key="select_matched_title")

        if st.button("이 책과 비슷한 도서 추천", use_container_width=True):
            # 선택 제목의 전역 인덱스 찾기
            target_title = st.session_state.select_matched_title
            # 전역 인덱스 (filtered 내)
            idx = None
            for i in st.session_state.matched_indices:
                if titles[i] == target_title:
                    idx = i
                    break
            if idx is None:
                st.error("선택한 책을 찾을 수 없습니다.")
            else:
                s_subj = cosine_similarity(X_subj[idx], X_subj).flatten()
                s_desc = cosine_similarity(X_desc[idx], X_desc).flatten()
                s_auth = cosine_similarity(X_auth[idx], X_auth).flatten()
                s_pub  = cosine_similarity(X_pub[idx],  X_pub ).flatten()

                content_sim = combine_content_score(s_subj, s_desc, s_auth, s_pub)
                final = final_score(content_sim, recency_vec)

                order = final.argsort()[::-1]
                recs = [i for i in order if i != idx][:top_n]

                st.write(f"**기준 도서:** {target_title}")
                if not recs:
                    st.info("추천 결과가 없습니다.")
                else:
                    for i in recs:
                        creator = to_text(raw_books[i].get("creator")) or "저자 정보 없음"
                        y = years[i] or "N/A"
                        p = pages[i] if pages[i] is not None else "N/A"
                        rel_keywords = pick_related_keywords(subjects_by_idx[i], picked_keywords=None, top_n=3)
                        kw_disp = " · 관련 키워드: " + ", ".join(rel_keywords) if rel_keywords else ""
                        st.markdown(
                            f"- **{titles[i]}** — {creator} (연도: {y}, 쪽수: {p})  "
                            f"· 콘텐츠점수: {content_sim[i]:.3f} · 최종점수: {final[i]:.3f}{kw_disp}"
                        )
    else:
        st.caption("검색 후 결과 목록에서 기준 도서를 선택하세요.")

# ---------- B) 키워드 검색형 ----------
with col2:
    st.subheader("📝 키워드 검색형 추천")
    st.caption("상위 10 키워드를 선택하거나, 자유 키워드를 입력하세요.")
    picked = st.multiselect("상위 키워드", options=top_keywords, default=[])
    q = st.text_input("자유 키워드 (선택)", placeholder="예: 도서관학, 저작권법, 역사")

    if st.button("키워드로 추천", use_container_width=True):
        if not picked and not q.strip():
            st.warning("키워드를 선택하거나 입력해 주세요.")
        else:
            query = " ".join(picked + ([q.strip()] if q.strip() else []))
            q_subj = vec_subj.transform([query])
            q_desc = vec_desc.transform([query])
            q_auth = vec_auth.transform([query])
            q_pub  = vec_pub.transform([query])

            s_subj = cosine_similarity(q_subj, X_subj).flatten()
            s_desc = cosine_similarity(q_desc, X_desc).flatten()
            s_auth = cosine_similarity(q_auth, X_auth).flatten()
            s_pub  = cosine_similarity(q_pub,  X_pub ).flatten()

            content_sim = combine_content_score(s_subj, s_desc, s_auth, s_pub)
            final = final_score(content_sim, recency_vec)

            order = final.argsort()[::-1][:top_n]
            st.write(f"**입력/선택 키워드:** {query}")
            for i in order:
                creator = to_text(raw_books[i].get("creator")) or "저자 정보 없음"
                y = years[i] or "N/A"
                p = pages[i] if pages[i] is not None else "N/A"
                rel_keywords = pick_related_keywords(subjects_by_idx[i], picked_keywords=picked, top_n=3)
                kw_disp = " · 관련 키워드: " + ", ".join(rel_keywords) if rel_keywords else ""
                st.markdown(
                    f"- **{titles[i]}** — {creator} (연도: {y}, 쪽수: {p})  "
                    f"· 콘텐츠점수: {content_sim[i]:.3f} · 최종점수: {final[i]:.3f}{kw_disp}"
                )
