# Streamlit 간단 앱: 생년월일(6자리) + 연도 선택 → 판별 결과 (요청 규칙 반영)
# 실행: pip install streamlit pandas numpy
#      streamlit run app.py

import re
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="소득세 판별", page_icon="🧾", layout="centered")
st.title("🧾 소득세 발급/신고 판별 (간단판)")

# ===== 데이터 로딩 =====
file = st.file_uploader("CSV 데이터 업로드", type=["csv"])
if not file:
    st.info("샘플 파일을 업로드해 주세요. (예: 세금데이터_2015-2024.csv)")
    st.stop()

df = pd.read_csv(file)

# 생년월일 컬럼 추정
def guess_birth_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "생년월일" in c]
    if not candidates:
        return df.columns[0]
    def score(col):
        return df[col].astype(str).str.fullmatch(r"\d{6}").fillna(False).mean()
    return max(candidates, key=score)

birth_col = guess_birth_col(df)

# (간단판) 연도 추출: 컬럼명에 4자리 연도(20xx)가 들어있는 경우만 매핑
year_map = {}
for c in df.columns:
    m = re.search(r"(20\d{2})", str(c))
    if m:
        year_map[int(m.group(1))] = c
years_available = sorted(year_map.keys())

birth6 = st.text_input("생년월일 6자리(YYMMDD)")
selected_years = st.multiselect(
    "조회 연도 선택 (복수 선택 가능)",
    years_available,
    default=years_available[-1:] if years_available else []
)

if st.button("판별하기", type="primary"):
    if not re.fullmatch(r"\d{6}", birth6 or ""):
        st.error("생년월일은 6자리 숫자로 입력하세요 (예: 900101)")
        st.stop()
    if not selected_years:
        st.error("연도를 최소 1개 선택하세요.")
        st.stop()

    # 대상자 행 추출
    person_rows = df[df[birth_col].astype(str).str.strip() == birth6]
    if person_rows.empty:
        st.warning("해당 생년월일과 일치하는 데이터가 없습니다.")
        st.stop()

    row = person_rows.iloc[0]

    def truthy(v):
        s = str(v).strip().upper()
        return s in {"1", "Y", "TRUE", "T", "예", "O", "YES", "제출"}

    # 선택 연도별 제출여부 수집
    submissions = {}
    for y in selected_years:
        col = year_map.get(y)
        submissions[y] = truthy(row[col]) if (col in row) else False

    # ① 모든 항목 N ?
    all_false = not any(submissions.values())

    # ② 기타소득/기타소득(간이)/연금계좌만 Y 인지 검사
    #    - '기타소득(간이)'도 '기타소득'에 포함되도록 처리
    is_gita = lambda name: ("기타소득" in name)
    is_pension = lambda name: ("연금계좌" in name)

    gita_y = any(truthy(row[c]) for c in row.index if is_gita(str(c)))
    pension_y = any(truthy(row[c]) for c in row.index if is_pension(str(c)))
    # '타 항목' = 위 두 분류가 아닌 모든 제출 컬럼
    other_y  = any(truthy(row[c]) for c in row.index if not (is_gita(str(c)) or is_pension(str(c))))

    # 최종 판별
    if all_false:
        result = "발급 가능"
    elif (gita_y and (not pension_y) and (not other_y)) or (pension_y and (not gita_y) and (not other_y)):
        result = "지급명세서 조회 필요"
    else:
        result = "발급 불가"

    st.success("판별 완료")
    st.write({
        "생년월일": birth6,
        "선택연도": selected_years,
        "연도별_제출여부": submissions,
        "기타소득만_Y": gita_y and not pension_y and not other_y,
        "연금계좌만_Y": pension_y and not gita_y and not other_y,
        "결과": result,
    })

st.caption("⚠️ 간단 규칙으로 동작합니다. 실제 데이터 구조(컬럼명/연도표기)와 세부 예외는 별도 검증이 필요합니다.")
