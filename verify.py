import streamlit as st
import json, re, datetime, os
from collections import Counter
import numpy as np
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# =========================
# 기본 세팅 & 스타일
# =========================
st.set_page_config(page_title="국립중앙도서관 기반 도서 추천 시스템", layout="wide")
st.title("📚 국립중앙도서관 기반 도서 추천 시스템")
st.caption("JSON 업로드 또는 공개 URL → 페이지/키워드 필터 → 책 선택형(검색) / 키워드형 추천 · 주제/설명/저자/출판사 가중치 + 출간일 최근 5년 가중치")

# 관련 키워드 칩(결과 표시용): 연한 분홍색
st.markdown("""
<style>
.keyword-row { margin: .25rem 0 .5rem 0; }
.keyword-chip {
  display:inline-block; padding: 4px 10px; margin: 3px 8px 3px 0;
  background:#fecdd3; color:#7a1330; border-radius:8px; font-size:0.85rem;
  line-height:1.2; font-weight:700; border:1px solid #fda4af;
}
</style>
""", unsafe_allow_html=True)

# =========================
# 안전 로더 / 유틸
# =========================
def safe_json_from_text(text: str):
    text = text.lstrip("\ufeff")  # BOM 제거
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(text.lstrip())
        return obj

def safe_load_json_file(path:str):
    with open(path, "r", encoding="utf-8-sig") as f:
        txt = f.read()
    return safe_json_from_text(txt)

def safe_load_json_uploaded(uploaded_file):
    txt = uploaded_file.read().decode("utf-8-sig")
    return safe_json_from_text(txt)

def safe_load_json_url(url: str, timeout=15):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    # 일부 호스팅은 text/json 헤더 없이 내려줄 수 있으므로 text로 처리
    return safe_json_from_text(r.text)

def to_text(v):
    if v is None: return ""
    if isinstance(v, str): return v
    if isinstance(v, (int, float, bool)): return str(v)
    if isinstance(v, list): return " ".join(to_text(x) for x in v)
    if isinstance(v, dict): return " ".join(to_text(x) for x in v.values())
    return str(v)

def to_list(v):
    if v is None: return []
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, str): s = x.strip(); 
            elif isinstance(x, dict): s = to_text(list(x.values())).strip()
            else: s = to_text(x).strip()
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
    for c in [to_text(book.get("issuedYear")),
              to_text(book.get("issued")),
              to_text(book.get("datePublished")),
              to_text(book.get("publicationDate"))]:
        m = YEAR_RE.search(c or "")
        if m:
            try: return int(m.group(0))
            except: pass
    return None

def extract_pages(book):
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
    d = max(now_year - year, 0)
    return (5 - d) / 5.0 if d <= 5 else 0.0

def pick_related_keywords(subjects, picked_keywords=None, top_n=3):
    subs = [s for s in subjects if s]
    if not subs: return []
    picked_set = set([k.strip() for k in (picked_keywords or []) if k.strip()])
    inter = [s for s in subs if s in picked_set]
    result = inter[:top_n]
    if len(result) < top_n:
        for s in subs:
            if s not in result:
                result.append(s)
            if len(result) >= top_n:
                break
    return result[:top_n]

def render_keywords_row(keywords):
    if not keywords: return ""
    chips = "".join(f'<span class="keyword-chip">{kw}</span>' for kw in keywords)
    return f'<div class="keyword-row">{chips}</div>'

# =========================
# 데이터 변환
# =========================
def build_records(data):
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
# 데이터 입력: 업로드 / 공개 URL / (옵션)로컬 샘플
# =========================
st.sidebar.header("📂 데이터 불러오기")

use_url = st.sidebar.checkbox("공개 URL에서 샘플 자동 로드", value=True)
sample_url = st.sidebar.text_input(
    "샘플 JSON 공개 URL",
    value="https://raw.githubusercontent.com/geeeeeh/library-sample/main/nlk_books_500_ko_diverse.json",
    help="GitHub raw, Dropbox 'dl=1', Google Drive 'uc?export=download&id=' 등 공개로 접근 가능한 URL을 넣어주세요."
)

uploaded = st.file_uploader("또는 JSON 직접 업로드 (.json)", type=["json"])

data = None

if uploaded is not None:
    try:
        data = safe_load_json_uploaded(uploaded)
        st.sidebar.success("업로드된 JSON을 불러왔습니다.")
    except Exception as e:
        st.sidebar.error(f"업로드 JSON 읽기 실패: {e}")

elif use_url and sample_url.strip():
    try:
        data = safe_load_json_url(sample_url.strip(), timeout=20)
        st.sidebar.success("공개 URL에서 샘플 JSON을 불러왔습니다.")
    except Exception as e:
        st.sidebar.error(f"URL 로드 실패: {e}")

# (선택) 같은 폴더의 로컬 샘플 파일 자동 탐지 — URL/업로드 실패 대비
if data is None:
    local_sample = "nlk_books_500_ko_diverse.json"
    if os.path.exists(local_sample):
        try:
            data = safe_load_json_file(local_sample)
            st.sidebar.info(f"로컬 샘플 사용: {local_sample}")
        except Exception as e:
            st.sidebar.error(f"로컬 샘플 읽기 실패: {e}")

if data is None:
    st.error("유효한 JSON 데이터를 불러오지 못했습니다. URL 또는 업로드를 확인해 주세요.")
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
    if p is None: return include_no_pages
    return page_range[0] <= p <= page_range[1]

filtered = [r for r in records if pass_page_filter(r["pages"])]
if not filtered:
    st.warning("⚠️ 페이지 필터 조건에 맞는 도서가 없습니다. 범위를 넓혀주세요.")
    st.stop()

# =========================
# 상위 10 키워드
# =========================
all_subjects = []
for r in filtered: all_subjects.extend(r["subjects"])
top_keywords = [kw for kw, _ in Counter([s for s in all_subjects if s]).most_common(10)]

# =========================
# 말뭉치(제목/KDC 제외)
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
# 가중치 UI
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

# ---------- A) 책 선택형 (검색) ----------
with col1:
    st.subheader("🔖 책 선택형 추천 (검색)")
    query_title = st.text_input("제목 검색어를 입력하세요 (부분일치 지원)", placeholder="예: 도서관학, 저작권, 디지털도서관")

    if "matched_indices" not in st.session_state:
        st.session_state.matched_indices = []

    if st.button("검색"):
        q = (query_title or "").strip().lower()
        if not q:
            st.warning("검색어를 입력하세요.")
            st.session_state.matched_indices = []
        else:
            matches = [i for i, t in enumerate(titles) if q in (t or "").lower()]
            if not matches:
                st.info("검색 결과가 없습니다. 다른 키워드로 시도해보세요.")
            st.session_state.matched_indices = matches

    if st.session_state.matched_indices:
        options = [titles[i] for i in st.session_state.matched_indices]
        sel_title = st.selectbox("검색 결과에서 기준 도서를 선택하세요", options=options, index=0, key="select_matched_title")

        if st.button("이 책과 비슷한 도서 추천", use_container_width=True):
            target_title = st.session_state.select_matched_title
            idx = None
            for i in st.session_state.matched_indices:
                if titles[i] == target_title:
                    idx = i; break
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
                        kw_html = render_keywords_row(rel_keywords)
                        st.markdown(
                            f"- **{titles[i]}** — {creator} (연도: {y}, 쪽수: {p})  "
                            f"· 콘텐츠점수: {content_sim[i]:.3f} · 최종점수: {final[i]:.3f}",
                            unsafe_allow_html=True
                        )
                        st.markdown(kw_html, unsafe_allow_html=True)
    else:
        st.caption("검색 후 결과 목록에서 기준 도서를 선택하세요.")

# ---------- B) 키워드 검색형 ----------
with col2:
    st.subheader("📝 키워드 검색형 추천")
    st.caption("상위 10 키워드를 선택하거나, 자유 키워드를 입력하세요.")
    picked = st.multiselect("상위 키워드", options=top_keywords, default=[])
    q = st.text_input("자유 키워드 (선택)", placeholder="예: 도서관학, 저작권법, 역사")

    if st.button("키워드로 추천", use_container_width=True):
        if not picked and not (q or "").strip():
            st.warning("키워드를 선택하거나 입력해 주세요.")
        else:
            query = " ".join(picked + ([q.strip()] if (q or "").strip() else []))
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
                kw_html = render_keywords_row(rel_keywords)
                st.markdown(
                    f"- **{titles[i]}** — {creator} (연도: {y}, 쪽수: {p})  "
                    f"· 콘텐츠점수: {content_sim[i]:.3f} · 최종점수: {final[i]:.3f}",
                    unsafe_allow_html=True
                )
                st.markdown(kw_html, unsafe_allow_html=True)
