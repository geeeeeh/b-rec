import streamlit as st
import json, re, datetime
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="도서 추천 시스템(세부 가중치)", layout="wide")
st.title("📚 국립중앙도서관 기반 도서 추천 시스템 — 세부 가중치 버전")
st.caption("JSON 업로드 → 페이지/키워드 필터 → 책 선택형/키워드형 추천 · 제목/주제/설명/저자/출판사/KDC 개별 가중치 + 최근성 가중치")

# ---------------------------
# 안전 로더 / 유틸
# ---------------------------
def safe_load_json(file):
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
    if v is None:
        return []
    if isinstance(v, list):
        out = []
        for x in v:
            if isinstance(x, str):
                if x.strip(): out.append(x.strip())
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
    ext = to_list(book.get("extent"))
    found = []
    for token in ext:
        for m in PAGES_RE.finditer(token):
            try: found.append(int(m.group(1)))
            except: pass
    return max(found) if found else None

def recency_weight(year, now_year=None):
    if year is None: return 0.0
    if now_year is None: now_year = datetime.date.today().year
    d = now_year - year
    d = max(d, 0)
    if d <= 5: return (5 - d) / 5.0
    return 0.0

# ---------------------------
# 데이터 변환
# ---------------------------
def build_records(data):
    if not isinstance(data, dict): return []
    books = data.get("@graph", [])
    if not isinstance(books, list) or not books: return []
    recs = []
    for bk in books:
        recs.append({
            "title": to_text(bk.get("title")) or "(제목 없음)",
            "subtitle": to_text(bk.get("remainderOfTitle")),
            "creator": to_text(bk.get("creator")),
            "subjects": to_list(bk.get("subject")),
            "desc": to_text(bk.get("description")),
            "series": to_text(bk.get("titleOfSeries")),
            "publisher": to_text(bk.get("publisher")),
            "kdc": to_text(bk.get("kdc")),
            "place": to_text(bk.get("publicationPlace")),
            "year": extract_year(bk),
            "pages": extract_pages(bk),
            "raw": bk
        })
    return recs

# ---------------------------
# 업로드
# ---------------------------
uploaded = st.file_uploader("도서정보 JSON 파일 업로드", type=["json"])
if not uploaded:
    st.info("📂 먼저 JSON 파일을 업로드하세요.")
    st.stop()

data = safe_load_json(uploaded)
if data is None: st.stop()
records = build_records(data)
if not records:
    st.warning("⚠️ '@graph' 내 도서 데이터를 찾지 못했습니다.")
    st.stop()

# ---------------------------
# 페이지 필터
# ---------------------------
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

# ---------------------------
# 상위 10 키워드 제시
# ---------------------------
all_subjects = []
for r in filtered: all_subjects.extend(r["subjects"])
top_keywords = [kw for kw, _ in Counter([s for s in all_subjects if s]).most_common(10)]

# ---------------------------
# 필드별 코퍼스 구성 (세부 가중치용)
# ---------------------------
titles = [r["title"] for r in filtered]
title_texts = [r["title"] + (" " + r["subtitle"] if r["subtitle"] else "") for r in filtered]
subject_texts = [" ".join(r["subjects"]) for r in filtered]
desc_texts = [r["desc"] for r in filtered]
author_texts = [r["creator"] for r in filtered]
publisher_texts = [r["publisher"] for r in filtered]
kdc_texts = [r["kdc"] for r in filtered]
years = [r["year"] for r in filtered]
pages = [r["pages"] for r in filtered]
raw_books = [r["raw"] for r in filtered]

# 벡터라이저(필드별)
vec_title = TfidfVectorizer()
vec_subj = TfidfVectorizer()
vec_desc = TfidfVectorizer()
vec_auth = TfidfVectorizer()
vec_pub  = TfidfVectorizer()
vec_kdc  = TfidfVectorizer()

X_title = vec_title.fit_transform(title_texts)
X_subj  = vec_subj.fit_transform(subject_texts)
X_desc  = vec_desc.fit_transform(desc_texts)
X_auth  = vec_auth.fit_transform(author_texts)
X_pub   = vec_pub.fit_transform(publisher_texts)
X_kdc   = vec_kdc.fit_transform(kdc_texts)

now_year = datetime.date.today().year
recency_vec = np.array([recency_weight(y, now_year) for y in years], dtype=float)

# ---------------------------
# 가중치 UI
# ---------------------------
st.sidebar.markdown("### ⚖️ 세부 가중치 (콘텐츠)")
w_title = st.sidebar.slider("제목 가중치", 0.0, 1.0, 0.30, 0.05)
w_subj  = st.sidebar.slider("주제(키워드) 가중치", 0.0, 1.0, 0.30, 0.05)
w_desc  = st.sidebar.slider("설명(요약) 가중치", 0.0, 1.0, 0.15, 0.05)
w_auth  = st.sidebar.slider("저자 가중치", 0.0, 1.0, 0.10, 0.05)
w_pub   = st.sidebar.slider("출판사 가중치", 0.0, 1.0, 0.05, 0.05)
w_kdc   = st.sidebar.slider("KDC 가중치", 0.0, 1.0, 0.10, 0.05)

w_sum = w_title + w_subj + w_desc + w_auth + w_pub + w_kdc
if w_sum == 0:  # 전부 0이면 기본값으로
    w_title, w_subj, w_desc, w_auth, w_pub, w_kdc = 0.3, 0.3, 0.15, 0.1, 0.05, 0.1
    w_sum = 1.0

# 정규화
w_title, w_subj, w_desc, w_auth, w_pub, w_kdc = [w / w_sum for w in [w_title, w_subj, w_desc, w_auth, w_pub, w_kdc]]

st.sidebar.markdown("### ⏱ 최근성 가중치")
w_recency = st.sidebar.slider("최근 5년 가중치 비율", 0.0, 0.8, 0.30, 0.05,
                              help="최종 점수 = (1-비율)*콘텐츠점수 + (비율)*최근성")

top_n = st.sidebar.slider("추천 개수 (Top N)", 3, 15, 5)

def combine_content_score(sim_t, sim_s, sim_d, sim_a, sim_p, sim_k):
    return (w_title*sim_t + w_subj*sim_s + w_desc*sim_d + w_auth*sim_a + w_pub*sim_p + w_kdc*sim_k)

def final_score(content_sim, rec_vec):
    return (1 - w_recency) * content_sim + w_recency * rec_vec

# ---------------------------
# 레이아웃
# ---------------------------
col1, col2 = st.columns(2, vertical_alignment="top")

# ========== A) 책 선택형 ==========
with col1:
    st.subheader("🔖 책 선택형 추천")
    sel_title = st.selectbox("추천 기준이 될 책을 선택하세요", options=titles, index=0)
    if st.button("이 책과 비슷한 도서 추천", use_container_width=True):
        idx = titles.index(sel_title)
        # 각 필드 유사도
        s_title = cosine_similarity(X_title[idx], X_title).flatten()
        s_subj  = cosine_similarity(X_subj[idx],  X_subj ).flatten()
        s_desc  = cosine_similarity(X_desc[idx],  X_desc ).flatten()
        s_auth  = cosine_similarity(X_auth[idx],  X_auth ).flatten()
        s_pub   = cosine_similarity(X_pub[idx],   X_pub  ).flatten()
        s_kdc   = cosine_similarity(X_kdc[idx],   X_kdc  ).flatten()

        content_sim = combine_content_score(s_title, s_subj, s_desc, s_auth, s_pub, s_kdc)
        final = final_score(content_sim, recency_vec)

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
                st.markdown(
                    f"- **{titles[i]}** — {creator} (KDC: {kdc}, 연도: {y}, 쪽수: {p})  "
                    f"· 콘텐츠점수: {content_sim[i]:.3f} · 최종점수: {final[i]:.3f}"
                )

# ========== B) 키워드 검색형 ==========
with col2:
    st.subheader("📝 키워드 검색형 추천")
    st.caption("상위 10 키워드를 선택하거나, 자유 키워드를 입력하세요.")
    picked = st.multiselect("상위 키워드", options=top_keywords, default=[])
    q = st.text_input("자유 키워드 (선택)", placeholder="예: 도서관학, 저작권법, 역사")

    if st.button("키워드로 추천", use_container_width=True):
        if not picked and not q.strip():
            st.warning("키워드를 선택하거나 입력해 주세요.")
        else:
            # 쿼리 텍스트 구성 (필드별 동일 쿼리 사용)
            query = " ".join(picked + ([q.strip()] if q.strip() else []))

            q_title = vec_title.transform([query])
            q_subj  = vec_subj.transform([query])
            q_desc  = vec_desc.transform([query])
            q_auth  = vec_auth.transform([query])
            q_pub   = vec_pub.transform([query])
            q_kdc   = vec_kdc.transform([query])

            s_title = cosine_similarity(q_title, X_title).flatten()
            s_subj  = cosine_similarity(q_subj,  X_subj ).flatten()
            s_desc  = cosine_similarity(q_desc,  X_desc ).flatten()
            s_auth  = cosine_similarity(q_auth,  X_auth ).flatten()
            s_pub   = cosine_similarity(q_pub,   X_pub  ).flatten()
            s_kdc   = cosine_similarity(q_kdc,   X_kdc  ).flatten()

            content_sim = combine_content_score(s_title, s_subj, s_desc, s_auth, s_pub, s_kdc)
            final = final_score(content_sim, recency_vec)

            order = final.argsort()[::-1][:top_n]
            st.write(f"**입력/선택 키워드:** {query}")
            for i in order:
                creator = to_text(raw_books[i].get("creator")) or "저자 정보 없음"
                kdc = to_text(raw_books[i].get("kdc")) or "N/A"
                y = years[i] or "N/A"
                p = pages[i] if pages[i] is not None else "N/A"
                st.markdown(
                    f"- **{titles[i]}** — {creator} (KDC: {kdc}, 연도: {y}, 쪽수: {p})  "
                    f"· 콘텐츠점수: {content_sim[i]:.3f} · 최종점수: {final[i]:.3f}"
                )
