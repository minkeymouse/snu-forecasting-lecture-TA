from nbbuild import md, code, write_nb

cells = [
    md(r"""
# 경제 분석 및 예측과 데이터 지능 실습5: 데이터 엔지니어링 응용

실습4에서 단면 데이터의 **기본기**(점검·결측·이상치·필터링)를 다뤘다면,
이번에는 실제 분석에서 자주 마주치는 **응용 작업**을 연습합니다.
여러 파일을 합치고, 그룹 단위로 상대값을 만들고, 보고용 교차표를 만들고, 텍스트·날짜에서 정보를 뽑아내는 작업입니다.

## 이번 실습에서 다루는 것

- 다양한 형식 읽기: `read_json`, `read_xml`, 중첩 구조의 `json_normalize`
- 테이블 결합: `merge`의 `how`/`on`/`validate`/`indicator`
- 고급 그룹 연산: `transform`(그룹 상대값) vs `agg` vs `filter`, 인덱스 정리(`set_index`/`reset_index`)
- 그룹 내 순위와 상위 N: `groupby().rank()`, 그룹별 `idxmax`
- 보고용 구조 변환: `pivot_table(margins)`, `crosstab(normalize)`
- 정규표현식 문자열 추출: `str.extract`
- 날짜·기간 계산: `datetime` 차이로 소요일 만들기, `resample`·`shift`로 시간 단위 집계
- 텍스트 단어 수: `str.split`으로 단어 수를 세고 그룹별로 비교

> 실습3(시계열)의 `melt`/`pivot`이 *시간축* 변환이었다면, 여기서는 *범주 × 범주* 보고표와
> *그룹 상대화*에 초점을 둡니다. 같은 함수라도 쓰는 목적이 다릅니다.

## 사용 데이터 (`../datasets/de/`)

| 파일 | 내용 |
|------|------|
| `basic1_json.json`, `basic1_xml.xml` | `basic1`과 같은 데이터의 JSON/XML 버전 |
| `basic1.csv`, `basic3.csv` | 성격유형(`f4`)으로 결합할 본 테이블 + 참조 테이블 |
| `purchase.csv` | 고객 100명의 세그먼트·연령대·카테고리·구매금액 |
| `sales_branch.csv` | 지점별 거래 100건 |
| `e-commerce.csv` | 주문/도착 시각이 있는 거래 30건 |
| `Titanic.csv` | 이름에서 호칭을 추출할 때 사용 |
| `hamspam.csv` | 메시지 2,000건(ham/spam) — 단어 수 세기에 사용 |

References:
- [pandas: Merge, join, concatenate](https://pandas.pydata.org/docs/user_guide/merging.html)
- [pandas: Group by (split-apply-combine)](https://pandas.pydata.org/docs/user_guide/groupby.html)
"""),
    code("""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
DE = "../datasets/de/"
"""),
    md(r"""
## 1) 다양한 형식 읽기 — JSON / XML

현장 데이터는 CSV만 오지 않습니다. API 응답은 보통 **JSON**, 공공데이터는 **XML**인 경우가 많습니다.

- `pd.read_json`: 열 지향 JSON을 바로 데이터프레임으로.
- `pd.read_xml`: XML의 반복 요소(`<row>`)를 행으로. `lxml`이 없으면 `parser="etree"`(파이썬 기본 파서)를 쓰면 됩니다.
- 빈 태그(`<f1/>`)는 결측(`NaN`)으로 들어옵니다 — CSV의 빈 칸과 동일하게 다루면 됩니다.
"""),
    code("""
js = pd.read_json(DE + "basic1_json.json")
xm = pd.read_xml(DE + "basic1_xml.xml", parser="etree")

# XML에는 <index> 요소가 그대로 열로 들어오므로 정리
xm = xm.drop(columns=["index"])

print("JSON shape:", js.shape, "| XML shape:", xm.shape)
print("XML의 f1 결측 수:", int(xm["f1"].isna().sum()))   # 빈 태그 → NaN
# 세 형식이 동일한 데이터인지 확인
csv = pd.read_csv(DE + "basic1.csv")
print("JSON == CSV ? ", js.shape == csv.shape and (js["id"] == csv["id"]).all())
js.head(3)
"""),
    md(r"""
**중첩 JSON 다루기.** API 응답은 종종 딕셔너리 안에 딕셔너리가 들어 있습니다.
이때는 `pd.json_normalize`로 평탄화합니다. `sep`으로 중첩 키를 이어 붙입니다.
"""),
    code("""
nested = [
    {"id": 1, "user": {"name": "Kim", "addr": {"city": "Seoul"}}, "amount": 1000},
    {"id": 2, "user": {"name": "Lee", "addr": {"city": "Busan"}}, "amount": 2500},
]
flat = pd.json_normalize(nested, sep="_")
display(flat)
"""),
    md(r"""
## 2) 테이블 결합 — `merge`

서로 다른 표를 **공통 키**로 이어 붙입니다. SQL의 JOIN과 같습니다.

- `how`: `inner`(교집합) · `left`(왼쪽 보존) · `right` · `outer`(합집합)
- `on`: 양쪽에 같은 이름의 키가 있을 때. 이름이 다르면 `left_on`/`right_on`.
- `validate="many_to_one"`: 관계를 강제 검증 — 의도치 않은 **행 폭증(중복 키)** 을 조기에 잡아줍니다.
- `indicator=True`: 각 행이 어디서 왔는지(`both`/`left_only`/`right_only`) 표시.

`basic3`은 성격유형(`f4`)별 보조 정보(`r1`, `r2`)를 담은 **참조 테이블**입니다.
`basic1`의 각 사람에게 이 정보를 붙입니다(다대일 결합).
"""),
    code("""
b1 = pd.read_csv(DE + "basic1.csv")
b3 = pd.read_csv(DE + "basic3.csv")
print("basic1:", b1.shape, "| basic3(참조):", b3.shape)

merged = b1.merge(b3, on="f4", how="left", validate="many_to_one", indicator=True)
print("결합 결과:", merged.shape)
print("매칭 상태:\\n", merged["_merge"].value_counts())
display(merged[["id", "f4", "r1", "r2"]].head())
"""),
    md(r"""
## 3) 고급 그룹 연산 — `transform` vs `agg` vs `filter`

`groupby` 뒤에 무엇을 붙이느냐로 결과 모양이 달라집니다.

| 메서드 | 결과 행 수 | 용도 |
|--------|-----------|------|
| `agg` | 그룹 수만큼 **축소** | 그룹 요약표 |
| `transform` | **원래 행 수 유지** | 그룹 통계량을 각 행에 되돌림 → 그룹 상대값 |
| `filter` | 조건 통과 그룹의 행만 | 그룹 단위 선별 |

`purchase`에서 **자기 세그먼트 평균 대비** 얼마나 높/낮게 샀는지(그룹 내 z-score)를 `transform`으로 만듭니다.
"""),
    code("""
pu = pd.read_csv(DE + "purchase.csv")
print("세그먼트:", pu["세그먼트"].unique().tolist())
print("카테고리:", pu["카테고리"].unique().tolist())

# (agg) 세그먼트 × 카테고리 요약
summary = pu.groupby(["세그먼트", "카테고리"], as_index=False).agg(
    n=("구매금액", "size"),
    mean_amt=("구매금액", "mean"),
    max_amt=("구매금액", "max"),
)
print("\\n[agg: 그룹 요약]")
display(summary.head())

# (transform) 그룹 내 z-score: 같은 세그먼트 안에서의 상대 위치
pu["amt_z_in_seg"] = pu.groupby("세그먼트")["구매금액"].transform(lambda s: (s - s.mean()) / s.std())
print("[transform: 세그먼트 내 z-score]")
display(pu[["고객ID", "세그먼트", "구매금액", "amt_z_in_seg"]].head())
"""),
    md(r"""
**미션 1.** 각 고객의 구매금액을 *자기 세그먼트* 기준으로 표준화(z-score)했을 때,
z-score가 **1보다 큰**(자기 세그먼트 평균보다 1표준편차 이상 많이 산) 고객 수를 구하라.
"""),
    code("""
print(int((pu["amt_z_in_seg"] > 1).sum()))
"""),
    md(r"""
### 잠깐 — 그룹 결과의 인덱스 정리 (`set_index` / `reset_index`)

위에서 `as_index=False`를 줬는데, 이게 무슨 뜻일까요? 여러 키로 `groupby`하면 그 키들이 결과의 **인덱스(MultiIndex)** 로 들어갑니다.

- `reset_index()`: 인덱스를 다시 보통 열로 되돌립니다. `as_index=False`는 바로 이 과정을 한 번에 해 주는 단축형입니다.
- `set_index(열)`: 반대로 특정 열을 인덱스로 올립니다.

그룹 결과를 평평한 표로 정리해 두면, 바로 다음에 배울 **순위·피벗·병합**에 그대로 이어 쓸 수 있습니다.
"""),
    code("""
multi = pu.groupby(["세그먼트", "연령대"])["구매금액"].mean()   # 키 2개 → MultiIndex
print("결과 인덱스 이름:", multi.index.names)

flat = multi.reset_index()                                      # 인덱스를 열로 되돌림
same = flat.equals(pu.groupby(["세그먼트", "연령대"], as_index=False)["구매금액"].mean())
print("reset_index() 결과가 as_index=False와 같은가:", same)
display(flat.head(3))
"""),
    md(r"""
## 4) 그룹 내 순위와 상위 N

"카테고리별 구매금액 상위 2명"처럼 **그룹마다 상위 N개**를 뽑는 작업입니다.

- `groupby(key)[col].rank(ascending=False, method="min")` 으로 그룹 내 순위를 매기고 필터링하거나,
- 정렬 후 `groupby(key).head(n)` 으로 간단히 뽑을 수 있습니다.
"""),
    code("""
# 방법 A: 그룹 내 순위 후 필터
pu["rank_in_cat"] = pu.groupby("카테고리")["구매금액"].rank(ascending=False, method="min")
top2_a = pu[pu["rank_in_cat"] <= 2].sort_values(["카테고리", "rank_in_cat"])

# 방법 B: 정렬 후 head
top2_b = (pu.sort_values("구매금액", ascending=False)
            .groupby("카테고리", as_index=False)
            .head(2))

print("두 방법의 행 수 일치:", len(top2_a) == len(top2_b))
display(top2_a[["카테고리", "고객ID", "구매금액", "rank_in_cat"]])
"""),
    md(r"""
## 5) 보고용 구조 변환 — `pivot_table` / `crosstab`

집계 결과를 **행 × 열 매트릭스**로 펼쳐 보고서 형태로 만듭니다.

- `pivot_table(..., margins=True)`: 행/열 합계(총계) 자동 추가.
- `crosstab(..., normalize="index")`: 빈도표를 **행 기준 비율**로 정규화 → 구성비 비교에 유용.
"""),
    code("""
# 연령대 × 카테고리 평균 구매금액 (+ 총계)
pivot = pu.pivot_table(index="연령대", columns="카테고리",
                       values="구매금액", aggfunc="mean", margins=True, margins_name="전체")
print("[연령대 × 카테고리 평균 구매금액]")
display(pivot.round(0))

# 세그먼트별 카테고리 '구성비'
comp = pd.crosstab(pu["세그먼트"], pu["카테고리"], normalize="index").round(3)
print("[세그먼트별 카테고리 구성비(행 합=1)]")
display(comp)
"""),
    md(r"""
## 6) 정규표현식으로 문자열 추출 — `str.extract`

`Titanic`의 `Name`은 `"성, 호칭. 이름"` 구조입니다. 정규표현식의 **그룹 `( )`** 으로 호칭만 뽑습니다.

- 패턴 `r",\s*([^.]+)\."` → 쉼표 뒤 공백 이후부터 마침표 전까지를 한 그룹으로 캡처.
- 호칭은 성별·결혼·신분 정보를 담고 있어 좋은 파생변수가 됩니다(실습5-2에서 활용 가능).
"""),
    code("""
tt = pd.read_csv(DE + "Titanic.csv")
tt["title"] = tt["Name"].str.extract(r",\\s*([^.]+)\\.")
print("[호칭 분포]")
display(tt["title"].value_counts())

# 희소 호칭은 'Rare'로 묶기 — 모델 입력에 자주 쓰는 정리 방식
common = ["Mr", "Miss", "Mrs", "Master"]
tt["title_grp"] = tt["title"].where(tt["title"].isin(common), "Rare")
print("[호칭별 생존율]")
display(tt.groupby("title_grp")["Survived"].agg(["count", "mean"]).round(3))
"""),
    md(r"""
## 7) 날짜·기간 계산

두 시점의 차이로 **소요 기간**을 만드는 작업입니다.
`datetime`끼리 빼면 `Timedelta`가 나오고, `.dt.days`로 일수를 얻습니다.

`e-commerce` 데이터로 **주문 → 도착 배송 소요일**을 계산합니다.
"""),
    code("""
ec = pd.read_csv(DE + "e-commerce.csv")
ec["OrderDate"] = pd.to_datetime(ec["OrderDate"])
ec["ArrivalDate"] = pd.to_datetime(ec["ArrivalDate"])
ec["delivery_days"] = (ec["ArrivalDate"] - ec["OrderDate"]).dt.days

print("[카테고리별 평균 배송 소요일]")
display(ec.groupby("Category")["delivery_days"].mean().round(2).sort_values())
print("전체 평균 배송일:", round(ec["delivery_days"].mean(), 2))
"""),
    md(r"""
**미션 2.** `e-commerce`에서 배송 소요일(`delivery_days`)이 **7일을 초과**하는 주문의 비율(%)을
소수 첫째 자리까지 구하라.
"""),
    code("""
rate = (ec["delivery_days"] > 7).mean() * 100
print(round(rate, 1))
"""),
    md(r"""
### 날짜를 인덱스로 — `resample` 과 `shift`

날짜 자체를 인덱스로 올리면(`set_index`, 바로 앞에서 배운 것) 시간 단위로 묶는 일이 한 줄로 됩니다.

- `resample("ME")`(월말)·`"W"`(주)·`"D"`(일) 같은 빈도로 다시 묶어 집계합니다.
  연·월을 일일이 뽑아 `groupby`하는 대신, 날짜 인덱스가 그 일을 대신해 줍니다.
- `shift(1)`은 한 시점 전 값을 끌어옵니다. 지금 값에서 빼면 **전월 대비 증감**이 됩니다.
"""),
    code("""
sb = pd.read_csv(DE + "sales_branch.csv")
sb["거래일"] = pd.to_datetime(sb["거래일"])

monthly = sb.set_index("거래일")["매출액"].resample("ME").sum()   # 월말 기준 매출 합계
report = pd.DataFrame({
    "매출합계": monthly.astype(int),
    "전월대비": monthly - monthly.shift(1),                       # 한 달 전과의 차이
})
print("[월별 매출과 전월 대비 증감]")
display(report)
"""),
    md(r"""
## 8) 텍스트에서 단어 수 세기

문장을 띄어쓰기로 나눠 **단어 수**를 세는 것은 텍스트 데이터의 기본 피처입니다.

- `str.split()`로 단어 리스트를 만들고 `str.len()`으로 개수를 셉니다.
- 글자 수는 문자열에 `str.len()`을 바로 적용합니다.
- 이렇게 만든 단어 수를 그룹(label)별로 평균 내어 비교할 수 있습니다.
"""),
    code("""
hs = pd.read_csv(DE + "hamspam.csv")
hs["n_words"] = hs["text"].str.split().str.len()   # 띄어쓰기 기준 단어 수
hs["n_chars"] = hs["text"].str.len()               # 글자 수(공백 포함)

print("[label별 평균 단어 수]")
display(hs.groupby("label")["n_words"].mean().round(2))
display(hs[["label", "text", "n_words", "n_chars"]].head(3))
"""),
    md(r"""
**미션 3.** 메시지의 평균 단어 수가 더 많은 `label`(ham 또는 spam)을 구하라.
"""),
    code("""
mean_words = hs.groupby("label")["n_words"].mean()
print(mean_words.idxmax())
"""),
    md(r"""
## 9) 종합문제

데이터를 다루다 보면 *분할 → 그룹 집계 → 병합 → 파생 → 선택*처럼 여러 단계를 이어서 처리하게 됩니다.
이번 실습에서 배운 `groupby`·`transform`·`merge`를 묶어 풉니다.

**종합문제 1.** `purchase`에서
1. 각 고객의 구매금액을 *자기 연령대* 기준으로 표준화한다(그룹 내 z-score).
2. 표준화 점수가 가장 높은 **상위 5명**의 *원래 구매금액 평균*을 정수로 구하라.
"""),
    code("""
cap = pd.read_csv(DE + "purchase.csv")
cap["z_in_age"] = cap.groupby("연령대")["구매금액"].transform(lambda s: (s - s.mean()) / s.std())
top5 = cap.nlargest(5, "z_in_age")
display(top5[["고객ID", "연령대", "구매금액", "z_in_age"]])
print("상위 5명 평균 구매금액:", int(top5["구매금액"].mean()))
"""),
    md(r"""
**종합문제 2.** (분할 → 집계 → 병합 → 차이) `purchase`에서 세그먼트를 둘로 나눠 비교합니다.

1. `세그먼트 == '일반'` 과 `'프리미엄'` 으로 데이터를 나눈다.
2. 각각 **연령대별 평균 구매금액**을 구한다.
3. 두 결과를 `merge`해 연령대별 **평균 구매금액 차이(절댓값)** 를 만들고, **차이가 가장 큰 연령대**를 구하라.

> *분할 → `groupby` → `merge` → 파생 → 정렬/선택* 은 두 집단을 비교할 때 자주 쓰는 흐름입니다.
"""),
    code("""
pu = pd.read_csv(DE + "purchase.csv")
normal = pu[pu["세그먼트"] == "일반"].groupby("연령대")["구매금액"].mean()
premium = pu[pu["세그먼트"] == "프리미엄"].groupby("연령대")["구매금액"].mean()

diff = pd.merge(normal.rename("일반"), premium.rename("프리미엄"), on="연령대")
diff["차이"] = (diff["프리미엄"] - diff["일반"]).abs()
display(diff.round(0))
print("평균 구매금액 차이가 가장 큰 연령대:", diff["차이"].idxmax())
"""),
    md(r"""
## 생각해보기

1. `merge`에서 `validate="many_to_one"`을 빼고 `basic3`에 중복 `f4`가 있었다면 어떤 문제가 생길까요?
2. `transform`으로 만든 그룹 내 z-score와, `agg`로 만든 그룹 평균을 다시 `merge`하는 방법은 결과가 같을까요? 어느 쪽이 간결한가요?
3. `crosstab(normalize="index")` 대신 `normalize="columns"`로 바꾸면 해석이 어떻게 달라지나요?
4. 호칭(`title`)을 `Rare`로 묶는 기준(빈도 임계값)을 바꾸면 모델 성능에 어떤 영향이 있을까요?
5. 배송 소요일을 *시간 단위*까지 반영하려면(`.dt.days` 대신) 어떻게 계산해야 할까요?
"""),
]

write_nb("practices/practice5_1.ipynb", cells)
