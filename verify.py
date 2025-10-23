import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="도서 추천 시스템", layout="centered")
st.title("📚 국립중앙도서관 기반 도서 추천 시스템")
st.write("국립중앙도서관 JSON 파일을 업로드하면, 유사한 책을 추천해드립니다.")

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
        # 처음 객체만 파싱 (raw_decode는 선두의 하나만 읽음)
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
# 3) 코퍼스/타이틀 생성
# ---------------------------
def build_corpus(data):
    """
    NLK JSON의 '@graph' 기준으로 각 도서의 텍스트를 구성.
    필드 타입이 제각각이어도 to_text로 안전 처리.
    """
    if not isinstance(data, dict):
        return [], [], []

    books = data.get("@graph", [])
    if not isinstance(books, list) or not books:
        return [], [], []

    corpus, titles, raw = [], [], []
    for book in books:
        # 필요 필드들을 유연하게 조합
        parts = [
            to_text(book.get("title")),
            to_text(book.get("remainderOfTitle")),
            to_text(book.get("creator")),
            to_text(book.get("dcterms:creator")),
            to_text(book.get("subject")),
            to_text(book.get("description")),
            to_text(book.get("titleOfSeries")),
            to_text(book.get("publisher")),
            to_text(book.get("kdc")),
            to_text(book.get("issuedYear")),
        ]
        text = " ".join(p for p in parts if p)
        corpus.append(text)
        titles.append(to_text(book.get("title")) or "(제목 없음)")
        raw.append(book)
    return corpus, titles, raw

# ---------------------------
# 앱 본문
# ---------------------------
uploaded_file = st.file_uploader("도서정보 JSON 파일을 업로드하세요", type=["json"])

if not uploaded_file:
    st.info("📂 먼저 JSON 파일을 업로드하세요.")
    st.stop()

data = safe_load_json(uploaded_file)
if data is None:
    st.stop()

corpus, titles, raw_books = build_corpus(data)
if not corpus:
    st.warning("⚠️ '@graph' 내 도서 데이터를 찾지 못했습니다.")
    st.stop()

# 선택 UI
selected_title = st.selectbox("추천을 원하는 책 제목을 선택하세요", options=titles)

if st.button("추천받기"):
    # TF-IDF 벡터화 + 유사도
    try:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform(corpus)
    except ValueError:
        st.error("코퍼스가 비어 있거나 유효한 텍스트가 없습니다.")
        st.stop()

    # 선택한 책 인덱스
    try:
        idx = titles.index(selected_title)
    except ValueError:
        st.error("선택한 책을 찾을 수 없습니다.")
        st.stop()

    sims = cosine_similarity(tfidf[idx], tfidf).flatten()
    # 본인 제외 상위 5권
    order = sims.argsort()[::-1]
    recs = [i for i in order if i != idx][:5]

    st.subheader(f"📖 '{selected_title}'와(과) 유사한 도서 추천")
    if not recs:
        st.write("추천 결과가 없습니다.")
    else:
        for i in recs:
            # 보조 표시용 메타
            meta_kdc = to_text(raw_books[i].get("kdc"))
            meta_creator = to_text(raw_books[i].get("creator"))
            st.markdown(
                f"- **{titles[i]}** — {meta_creator or '저자 정보 없음'}  "
                f"(KDC: {meta_kdc or 'N/A'}, 유사도: {sims[i]:.3f})"
            )
