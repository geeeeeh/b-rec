# Streamlit 앱: CSV 업로드 → 생년월일·연도 선택 → 판별 + 근거 표시 (from-scratch)
# 설치: pip install streamlit pandas numpy
# 실행: streamlit run app.py

import re
import pandas as pd
import streamlit as st

st.set_page_config(page_title="소득세 판별", page_icon="🧾", layout="centered")
st.title("🧾 소득세 발급 판별 (근거 포함)")

# --- 파일 업로드 ---
file = st.file_uploader("CSV 데이터 업로드", type=["csv"]) 
if not file:
    st.info("샘플 파일을 업로드해 주세요. (예: 세금데이터_2015-2024.csv)")
    st.stop()

df = pd.read_csv(file)

# --- 생년월일 컬럼 자동 추정 ---
def guess_birth_col(df: pd.DataFrame) -> str:
    candidates = [c for c in df.columns if "생년월일" in c]
    if not candidates:
        return df.columns[0]
    def score(col):
        return df[col].astype(str).str.fullmatch(r"\d{6}").fillna(False).mean()
    return max(candidates, key=score)

birth_col = guess_birth_col(df)

# --- 연도 매핑: 위치(D~O) + 파일명 범위 감지 + 시작연도 입력(백업) ---
cols = list(df.columns)
position_cols = cols[3:3+12]  # D~O 최대 12개

fname = getattr(file, "name", "")
m = re.search(r"(20\d{2})\D+(20\d{2})", fname)
if m:
    y1, y2 = int(m.group(1)), int(m.group(2))
    seq = list(range(min(y1, y2), max(y1, y2)+1))
    years_seq = seq[:len(position_cols)]
else:
    start_year = st.number_input("시작연도 입력 (D열에 해당)", min_value=1990, max_value=2100, value=2015, step=1)
    years_seq = [int(start_year) + i for i in range(len(position_cols))]

# 길이 불일치 안전 매핑
year_map = dict(zip(years_seq, position_cols))  # {연도:int → 컬럼명:str}
if not year_map:
    st.error("연도 매핑 실패: 파일명 범위 표기 또는 시작연도 입력을 확인하세요.")
    st.stop()
inv_map = {v: k for k, v in year_map.items()}  # {컬럼명:str → 연도:int}

# --- 입력 ---
birth6 = st.text_input("생년월일 6자리(YYMMDD)")
years_available = sorted(year_map.keys())
selected_years = st.multiselect(
    "조회 연도 선택 (복수 선택 가능)", years_available,
    default=years_available[-1:] if years_available else []
)

# --- 유틸 ---
def truthy(v) -> bool:
    s = str(v).strip().upper()
    return s in {"1","Y","TRUE","T","예","O","YES","제출"}

def in_selected(col_name: str) -> bool:
    y = inv_map.get(col_name)
    return (y in selected_years) if y is not None else False

if st.button("판별하기", type="primary"):
    if not re.fullmatch(r"\d{6}", birth6 or ""):
        st.error("생년월일은 6자리 숫자(YYMMDD)로 입력하세요. 예: 900101")
        st.stop()
    if not selected_years:
        st.error("연도를 최소 1개 선택하세요.")
        st.stop()

    # 대상자 찾기
    person_rows = df[df[birth_col].astype(str).str.strip() == birth6]
    if person_rows.empty:
        st.warning("해당 생년월일과 일치하는 데이터가 없습니다.")
        st.stop()
    row = person_rows.iloc[0]

    # 선택연도 제출여부 (연도 → True/False)
    submissions = {}
    for y in selected_years:
        col = year_map.get(y)
        submissions[y] = truthy(row[col]) if (col in row) else False

    # 선택연도 내 카테고리별 True 컬럼 목록 수집(근거)
    def is_gita(name: str) -> bool:
        return "기타소득" in name  # '기타소득(간이)' 포함
    def is_pension(name: str) -> bool:
        return "연금계좌" in name

    gita_true_cols = [str(c) for c in row.index if in_selected(str(c)) and is_gita(str(c)) and truthy(row[c])]
    pension_true_cols = [str(c) for c in row.index if in_selected(str(c)) and is_pension(str(c)) and truthy(row[c])]
    other_true_cols = [str(c) for c in row.index if in_selected(str(c)) and (not (is_gita(str(c)) or is_pension(str(c)))) and truthy(row[c])]

    # 요약 플래그
    all_false = not any(submissions.values())
    gita_only = (len(gita_true_cols) > 0) and (len(pension_true_cols) == 0) and (len(other_true_cols) == 0)
    pension_only = (len(pension_true_cols) > 0) and (len(gita_true_cols) == 0) and (len(other_true_cols) == 0)

    # 최종 판정
    if all_false:
        result = "발급 가능"
        reason = "선택한 연도에 제출(Y) 항목이 없습니다."
    elif gita_only or pension_only:
        result = "지급명세서 조회 필요"
        if gita_only:
            reason = "선택한 연도에서 '기타소득' 관련 항목만 Y이고, 다른 항목은 모두 N입니다."
        else:
            reason = "선택한 연도에서 '연금계좌' 관련 항목만 Y이고, 다른 항목은 모두 N입니다."
    else:
        result = "발급 불가"
        reason = "선택한 연도에 Y가 여러 항목에서 확인되어 단일 요건을 충족하지 않습니다."

    # 출력
    st.success("판별 완료")
    st.subheader("결과")
    st.write({
        "생년월일": birth6,
        "선택연도": selected_years,
        "결과": result,
        "사유": reason,
    })

    st.subheader("근거")
    st.write({
        "연도별_제출여부(선택연도)": submissions,
        "Y_기타소득_컬럼(선택연도)": gita_true_cols or ["없음"],
        "Y_연금계좌_컬럼(선택연도)": pension_true_cols or ["없음"],
        "Y_그외_컬럼(선택연도)": other_true_cols or ["없음"],
    })

st.caption("⚠️ 위치(D~O) 기반으로 연도 매핑을 수행하고, 선택한 연도 범위 내 컬럼만 판정·근거 수집에 사용합니다. 파일 구조가 다르면 시작연도로 조정하세요.")
