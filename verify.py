import streamlit as st
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="도서 추천 시스템", layout="centered")

st.title("📚 국립중앙도서관 기반 도서 추천 시스템")
st.write("국립중앙도서관 JSON 파일을 업로드하면, 유사한 책을 추천해드립니다.")

# 1️⃣ JSON 파일 업로드
uploaded_file = st.file_uploader("도서정보 JSON 파일을 업로드하세요", type=["json"])

if uploaded_file:
    data = json.load(uploaded_file)
    books = data.get("@graph", [])

    if not books:
        st.warning("⚠️ JSON 구조에 '@graph' 항목이 없습니다.")
    else:
        # 2️⃣ 도서 텍스트 데이터 준비
        corpus, titles = [], []
        for book in books:
            text = " ".join([
                book.get("title", ""),
                str(book.get("creator", "")),
                " ".join(book.get("subject", [])),
                book.get("description", ""),
                book.get("kdc", "")
            ])
            corpus.append(text)
            titles.append(book.get("title", ""))

        # 3️⃣ 책 선택
        selected_title = st.selectbox("추천을 원하는 책 제목을 선택하세요", titles)

        if st.button("추천받기"):
            # 4️⃣ TF-IDF 유사도 계산
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(corpus)

            try:
                idx = titles.index(selected_title)
            except ValueError:
                st.error("선택한 책을 찾을 수 없습니다.")
                st.stop()

            cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
            similar_indices = cosine_sim.argsort()[-6:][::-1]

            st.subheader(f"📖 '{selected_title}'와(과) 유사한 도서 추천 결과")

            for i in similar_indices[1:]:
                st.write(f"- **{titles[i]}** (유사도: {cosine_sim[i]:.3f})")

else:
    st.info("📂 먼저 JSON 파일을 업로드하세요.")
