# Streamlit 간단 앱: 생년월일(6자리) + 연도 선택 → 판별 결과 (위치매핑+특수케이스 반영)
# 실행: pip install streamlit pandas numpy
#      streamlit run app.py

import re
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="소득세 판별", page_icon="🧾", layout="centered")
st.title("🧾 소득세 발급/신고 판별 (간단판)")

# 신고대상 판단(자리표시자: 현재는 항상 False) — 실제 법령 로직은 추후 반영

def is_reportable(row: pd.Series, years: list[int]) -> bool:
    return False

# ===== 데이터 로딩 =====
file = st.file_uploader("CSV 데이터 업로드", type=["csv"]) 
if not file:
    st.info("샘플 파일을 업로드해 주세요. (예: 세금데이터_2015-2024.csv)")
    st.stop()

df = pd.read_csv(file)

# 생년월일 컬럼 추정 (이름에 '생년월일' 포함 & 길이 6인 값 비율이 높은 컬럼)

def guess_birth_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "생년월일" in c]
    if not candidates:
        return df.columns[0]
    def score(col):
        s = df[col].astype(str).str.fullmatch(r"\d{6}").fillna(False).mean()
        return s
    return max(candidates, key=score)

birth_col = guess_birth_col(df)

# ===== 가장 빠른 연도 매핑: 위치(D~O) 기반 + 파일명 자동 감지 + zip 안전 매핑 =====
col_names = list(df.columns)
position_cols = col_names[3:3+12]  # D~O (최대 12개)

# 파일명에서 연도 범위 감지 (예: 세금데이터_2015-2024.csv)
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
    st.error("연도 매핑에 실패했습니다. 시작연도를 입력하거나 파일명을 '2015-2024' 형식으로 지정해 주세요.")
    st.stop()

years_available = sorted(year_map.keys())

birth6 = st.text_input("생년월일 6자리(YYMMDD)")
selected_years = st.multiselect("조회 연도 선택 (복수 선택 가능)", years_available, default=years_available[-1:] if years_available else [])

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

    # 제출여부 판독(Y/1/True/예/YES/제출 등 수용)
    def truthy(v):
        s = str(v).strip().upper()
        return s in {"1", "Y", "TRUE", "T", "예", "O", "YES", "제출"}

    # 선택 연도별 제출여부
    submissions = {}
    for y in selected_years:
        col = year_map.get(y)
        submissions[y] = truthy(row[col]) if (col in row) else False

    any_submitted = any(submissions.values())

    # 간이 규칙: 지급명세서 'Y'가 하나라도 있으면 소득 존재로 간주
    income_exists = any_submitted

    # === 특수 케이스 ===
    # 전부 N(= any_submitted False) 이지만,
    # '기타소득', '기타소득(간이)', '연금계좌' 중 "하나만" Y인 경우 → "발급가능(지급명세서 조회 필요)"
    special_keywords = ["기타소득(간이)", "기타소득", "연금계좌"]
    special_y_count = 0
    for col in row.index:
        name_col = str(col)
        if any(k in name_col for k in special_keywords) and truthy(row[col]):
            special_y_count += 1
    only_special_one = (not any_submitted) and (special_y_count == 1)

    # 신고대상 여부(현 단계: 자리표시자)
    reportable = is_reportable(row, selected_years)

    # 최종 판별 (요청 규칙 반영)
    # 1) 신고대상(True) & 제출 없음 → "신고 필요"
    # 2) 제출(Y) 하나라도 있음 → "타 증명 발급 필요"
    # 3) 제출 전부 N + (특수케이스 단일 Y) → "발급가능(지급명세서 조회 필요)"
    # 4) 그 외 → "발급가능"
    if reportable and not any_submitted:
        result = "신고 필요"
    elif any_submitted:
        result = "타 증명 발급 필요"
    elif only_special_one:
        result = "발급가능(지급명세서 조회 필요)"
    else:
        result = "발급가능"

    st.success("판별 완료")
    st.write({
        "생년월일": birth6,
        "선택연도": selected_years,
        "연도별_제출여부": submissions,
        "신고대상여부(가정)": reportable,
        "특수케이스_단일_Y": only_special_one,
        "소득존재여부(간이)": income_exists,
        "결과": result,
    })

st.caption("⚠️ 위치(D~O) 기반 매핑을 사용합니다. 파일 구조가 달라질 경우 시작연도 입력칸으로 조정하세요. 실제 법령 기반 신고대상 로직은 후속 단계에서 반영이 필요합니다.")
