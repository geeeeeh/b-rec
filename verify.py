# Streamlit 앱: CSV 업로드 → 장르·키워드 선택 → 책 추천
# 설치: pip install streamlit pandas numpy
# 실행: streamlit run book_recommender.py

import re
import pandas as pd
import streamlit as st
import numpy as np

st.set_page_config(page_title="책 추천", page_icon="📚", layout="wide")
st.title("📚 개인 맞춤 책 추천 시스템")

# --- 파일 업로드 ---
file = st.file_uploader("책 데이터 CSV 업로드", type=["csv"]) 
if not file:
    st.info("📖 책 데이터 CSV 파일을 업로드해 주세요.")
    st.markdown("""
    **CSV 파일 형식 예시:**
    - 필수 컬럼: 제목, 저자, 장르
    - 선택 컬럼: 출판연도, 평점, 페이지수, 키워드, 설명
    """)
    st.stop()

df = pd.read_csv(file)

# --- 컬럼 자동 추정 ---
def guess_column(df: pd.DataFrame, keywords: list) -> str:
    """키워드 리스트로 컬럼명 추정"""
    for keyword in keywords:
        candidates = [c for c in df.columns if keyword in c.lower()]
        if candidates:
            return candidates[0]
    return df.columns[0] if len(df.columns) > 0 else None

title_col = guess_column(df, ["제목", "title", "책"])
author_col = guess_column(df, ["저자", "author", "작가"])
genre_col = guess_column(df, ["장르", "genre", "분류"])
year_col = guess_column(df, ["연도", "year", "출판"])
rating_col = guess_column(df, ["평점", "rating", "점수"])
keyword_col = guess_column(df, ["키워드", "keyword", "태그"])

# 컬럼 확인
st.sidebar.header("📋 데이터 정보")
st.sidebar.write(f"총 {len(df)}권의 책")
st.sidebar.write(f"컬럼 수: {len(df.columns)}")

with st.sidebar.expander("컬럼 매핑 확인"):
    st.write(f"제목: {title_col}")
    st.write(f"저자: {author_col}")
    st.write(f"장르: {genre_col}")
    st.write(f"출판연도: {year_col}")
    st.write(f"평점: {rating_col}")
    st.write(f"키워드: {keyword_col}")

# --- 사용자 입력 ---
st.header("🎯 선호도 설정")

col1, col2 = st.columns(2)

with col1:
    # 장르 선택
    if genre_col and genre_col in df.columns:
        unique_genres = df[genre_col].dropna().unique().tolist()
        selected_genres = st.multiselect(
            "선호 장르 선택 (복수 선택 가능)",
            options=unique_genres,
            default=[]
        )
    else:
        selected_genres = []
        st.warning("장르 정보를 찾을 수 없습니다.")

    # 출판연도 범위
    if year_col and year_col in df.columns:
        years = pd.to_numeric(df[year_col], errors='coerce').dropna()
        if len(years) > 0:
            min_year = int(years.min())
            max_year = int(years.max())
            year_range = st.slider(
                "출판연도 범위",
                min_value=min_year,
                max_value=max_year,
                value=(min_year, max_year)
            )
        else:
            year_range = None
    else:
        year_range = None

with col2:
    # 키워드 검색
    if keyword_col and keyword_col in df.columns:
        search_keyword = st.text_input("키워드 검색 (선택사항)", "")
    else:
        search_keyword = ""
    
    # 평점 필터
    if rating_col and rating_col in df.columns:
        ratings = pd.to_numeric(df[rating_col], errors='coerce').dropna()
        if len(ratings) > 0:
            min_rating = st.slider(
                "최소 평점",
                min_value=0.0,
                max_value=5.0,
                value=0.0,
                step=0.5
            )
        else:
            min_rating = 0.0
    else:
        min_rating = 0.0

    # 추천 개수
    num_recommendations = st.number_input(
        "추천받을 책 개수",
        min_value=1,
        max_value=50,
        value=5,
        step=1
    )

# --- 추천 로직 ---
if st.button("📚 책 추천받기", type="primary"):
    
    filtered_df = df.copy()
    
    # 장르 필터
    if selected_genres and genre_col in df.columns:
        filtered_df = filtered_df[filtered_df[genre_col].isin(selected_genres)]
    
    # 연도 필터
    if year_range and year_col in df.columns:
        filtered_df[year_col] = pd.to_numeric(filtered_df[year_col], errors='coerce')
        filtered_df = filtered_df[
            (filtered_df[year_col] >= year_range[0]) & 
            (filtered_df[year_col] <= year_range[1])
        ]
    
    # 키워드 필터
    if search_keyword and keyword_col in df.columns:
        filtered_df = filtered_df[
            filtered_df[keyword_col].astype(str).str.contains(search_keyword, case=False, na=False)
        ]
    
    # 평점 필터
    if min_rating > 0.0 and rating_col in df.columns:
        filtered_df[rating_col] = pd.to_numeric(filtered_df[rating_col], errors='coerce')
        filtered_df = filtered_df[filtered_df[rating_col] >= min_rating]
    
    # 결과 확인
    if filtered_df.empty:
        st.error("😢 조건에 맞는 책을 찾을 수 없습니다. 필터 조건을 조정해보세요.")
        st.stop()
    
    # 평점 기준 정렬 (있는 경우)
    if rating_col in filtered_df.columns:
        filtered_df = filtered_df.sort_values(by=rating_col, ascending=False, na_position='last')
    
    # 추천 개수만큼 선택
    recommendations = filtered_df.head(num_recommendations)
    
    # 결과 출력
    st.success(f"✨ {len(recommendations)}권의 책을 추천합니다!")
    
    st.subheader("📖 추천 도서 목록")
    
    for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
        with st.expander(f"#{idx} {row[title_col]}", expanded=(idx <= 3)):
            cols = st.columns([2, 1])
            
            with cols[0]:
                st.markdown(f"**제목:** {row[title_col]}")
                if author_col in row:
                    st.markdown(f"**저자:** {row[author_col]}")
                if genre_col in row:
                    st.markdown(f"**장르:** {row[genre_col]}")
            
            with cols[1]:
                if year_col in row and pd.notna(row[year_col]):
                    st.metric("출판연도", f"{int(row[year_col])}년")
                if rating_col in row and pd.notna(row[rating_col]):
                    st.metric("평점", f"⭐ {row[rating_col]}")
            
            if keyword_col in row and pd.notna(row[keyword_col]):
                st.markdown(f"🏷️ **키워드:** {row[keyword_col]}")
            
            # 설명이 있다면 표시
            desc_cols = [c for c in row.index if any(k in c.lower() for k in ["설명", "desc", "소개", "summary"])]
            if desc_cols and pd.notna(row[desc_cols[0]]):
                st.markdown(f"📝 {row[desc_cols[0]]}")
    
    # 필터링 통계
    st.divider()
    st.subheader("📊 필터링 통계")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("전체 책", f"{len(df)}권")
    with col2:
        st.metric("조건 충족", f"{len(filtered_df)}권")
    with col3:
        st.metric("추천", f"{len(recommendations)}권")
    
    # 상세 데이터 테이블
    with st.expander("📋 추천 도서 상세 데이터"):
        st.dataframe(recommendations, use_container_width=True)

st.caption("💡 Tip: 장르나 키워드를 선택하지 않으면 전체 책에서 추천합니다. 평점이 높은 순으로 추천됩니다.")
