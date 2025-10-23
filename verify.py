# Streamlit 간단 앱: 생년월일(6자리) + 연도 선택 → 판별 결과 (위치매핑 + 새 판별 로직)
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

# ===== 연도 추출 (개선: 위치 D~O 기반 + 파일명 범위 자동감지 + 시작연도 입력) =====
col_names = list(df.columns)
position_cols = col_names[3:3+12]  # D~O(최대 12개)

name = getattr(file, "name", "")
m = re.search(r"(20\d{2})\D+(20\d{2})", name)
if m:
    y1, y2 = int(m.group(1)), int(m.group(2))
    seq = list(range(min(y1, y2), max(y1, y2)+1))
    years_seq = seq[:len(position_cols)]
else:
    start_year = st.number_input("시작연도 입력 (D열에 해당)", min_value=1990, max_value=2100, value=2015, step=1)
    years_seq = [int(start_year) + i for i in range(len(position_cols))]

# 길이 불일치 안전 매핑
year_map = dict(zip(years_seq, position_cols))
if not year_map:
    st.error("연도 매핑에 실패했습니다. 파일명을 '2015-2024' 형식으로 지정하거나 시작연도를 입력해 주세요.")
    st.stop()

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

    # 선택 연도 범위 안에서만 카테고리별 Y 여부 판단
    inv_map = {v: k for k, v in year_map.items()}
    def in_selected(col_name: str) -> bool:
        y = inv_map.get(col_name)
        return (y in selected_years) if y is not None else False

    is_gita = lambda name: ("기타소득" in name)          # '기타소득(간이)' 포함
    is_pension = lambda name: ("연금계좌" in name)

    gita_y = any(in_selected(str(c)) and is_gita(str(c)) and truthy(row[c]) for c in row.index)
    pension_y = any(in_selected(str(c)) and is_pension(str(c)) and truthy(row[c]) for c in row.index)
    other_y = any(in_selected(str(c)) and (not (is_gita(str(c)) or is_pension(str(c)))) and truthy(row[c]) for c in row.index)

    # === 새 판별 로직 ===
    # 모든 항목 N → '발급 가능'
    # 타 항목 N 이면서 (기타소득만 Y 또는 연금계좌만 Y) → '지급명세서 조회 필요'
    # 그 외 모든 경우 → '발급 불가'
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
        "기타소득만_Y": gita_y and (not pension_y) and (not other_y),
        "연금계좌만_Y": pension_y and (not gita_y) and (not other_y),
        "결과": result,
    })

st.caption("⚠️ 위치(D~O) 기반 매핑을 사용합니다. 파일 구조가 달라질 경우 시작연도 입력칸으로 조정하세요. "
           "실제 세법 기준·예외는 별도로 반영해야 합니다.")
