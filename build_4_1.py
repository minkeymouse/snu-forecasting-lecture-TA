from nbbuild import md, code, write_nb

cells = [
    md(r"""
# 경제 분석 및 예측과 데이터 지능 실습4: 데이터 엔지니어링 일반

이번 실습은 모델링에 들어가기 전, **표(table) 형태의 단면(cross-sectional) 데이터를 다루는 기본기**를 정리합니다.
실습 3에서 다룬 *시계열* 데이터 엔지니어링(리샘플링·lag·melt/pivot)과 달리, 여기서는 한 행이 하나의 관측치인
일반적인 데이터프레임을 대상으로 **적재 → 점검 → 정제 → 변형 → 집계**의 흐름을 연습합니다.

## 이번 실습에서 다루는 것

- 데이터 적재와 첫 점검: `info`, `describe`, 결측·중복·자료형 확인
- 자료형 정리: `astype`, `category`, 메모리 관리
- 결측값 처리: 단순 대치 vs **그룹별 대치**, 최빈값 대치
- 이상치 탐지: **IQR 규칙**과 표준편차(z-score) 규칙
- 스케일링과 인코딩: `StandardScaler`·`MinMaxScaler`·`RobustScaler`, `get_dummies`·라벨 인코딩
- 조건 필터링과 정렬: 불리언 인덱싱, `query`, `nlargest`
- 구간화와 순위: `cut`, `qcut`, `rank`
- 문자열·날짜 처리: `.str` 접근자, `to_datetime`, `.dt` 성분
- 연·월 집계와 상관관계: `groupby`로 월별 합계, `corr`로 변수 간 관계

## 진행 방식

각 주제는 **개념 정리 → 미션(문제) → 풀이** 순서로 구성됩니다.
미션은 "조건을 만족하는 값 하나를 구해 출력"하는 형태로, 전처리 결과를 숫자 한 개로 검증하는 연습입니다.

## 사용 데이터 (`../datasets/de/`)

| 파일 | 단위 | 비고 |
|------|------|------|
| `basic1.csv` | 개인 100명 | 결측(`f1`,`f3`)·범주형(`city`,`f4`)이 섞인 연습용 표 |
| `basic2.csv` | 일별 730행 | `Sales`,`PV`,`UV`,`Events` — 이상치·날짜 처리용 |
| `Titanic.csv` | 승객 891명 | 필터링·구간화·문자열 처리용 (실습5-2에서 모델링에 재사용) |

References:
- [pandas: Working with missing data](https://pandas.pydata.org/docs/user_guide/missing_data.html)
- [scikit-learn: Preprocessing data](https://scikit-learn.org/stable/modules/preprocessing.html)
"""),
    code("""
import pandas as pd
import numpy as np

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

DE = "../datasets/de/"
print("pandas", pd.__version__)
"""),
    md(r"""
## 0) 데이터 적재와 첫 점검

새 데이터를 받으면 **모델링 코드를 짜기 전에** 반드시 형태를 파악합니다. 최소 점검 항목은 다음과 같습니다.

- `shape`: 행/열 개수 (관측치 수, 변수 수)
- `dtypes`: 각 열의 자료형 (숫자인데 문자로 들어와 있지 않은가?)
- `isna()`: 결측이 어디에 얼마나 있는가?
- `describe()`: 수치형의 분포(평균·분위수·최댓값)와 이상값 신호

`describe(include="all")`를 쓰면 범주형의 최빈값·고유값 개수까지 한 번에 볼 수 있습니다.
"""),
    code("""
df = pd.read_csv(DE + "basic1.csv")

print("shape:", df.shape)
print("\\n[자료형]")
print(df.dtypes)
df.head()
"""),
    code("""
print("[결측 개수]")
print(df.isna().sum())
print("\\n[결측 비율(%)]")
print((df.isna().mean() * 100).round(1))

print("\\n[수치형 요약]")
display(df.describe().T)
print("\\n[범주형 요약]")
display(df.describe(include=["object", "string"]).T)
"""),
    md(r"""
### 중복 행 확인과 제거

점검 단계에서 빼놓을 수 없는 것이 **중복 행**입니다. 같은 행이 여러 번 들어오면 집계가 부풀려집니다.
`duplicated()`로 확인하고 `drop_duplicates()`로 제거하며, 특정 열 기준만 보려면 `subset=`을 줍니다.
"""),
    code("""
print("완전히 같은 행 수:", int(df.duplicated().sum()))
print("city+f4 조합 중복 행 수:", int(df.duplicated(subset=["city", "f4"]).sum()))

# 인위적으로 중복을 만들어 제거를 확인 (pd.concat으로 위 2행을 복제)
dup_demo = pd.concat([df.head(2), df.head(2)], ignore_index=True)
print("중복 생성 후:", len(dup_demo), "→ drop_duplicates 후:", len(dup_demo.drop_duplicates()))
"""),
    md(r"""
## 1) 자료형 정리

자료형은 단순한 표기 문제가 아니라 **연산의 의미와 메모리**를 좌우합니다.

- `f2`는 `0/1/2` 코드값입니다. 평균을 내는 건 의미가 없으므로 **범주형(`category`)** 으로 두는 편이 안전합니다.
- `age`가 실수(`2.0`, `9.0`)로 저장돼 있지만 정수 개념이므로 정수형으로 바꿉니다.
- 범주형으로 바꾸면 메모리도 줄어듭니다.

> 결측이 있는 정수 열을 정수형으로 바꿀 때는 일반 `int`가 아니라 결측 허용 정수형 `"Int64"`를 씁니다.
> (`age`에는 결측이 없으므로 여기서는 `int64`로 충분합니다.)
"""),
    code("""
work = df.copy()
work["age"] = work["age"].astype("int64")     # 실수 → 정수
work["f2"] = work["f2"].astype("category")     # 코드값 → 범주형

print(work.dtypes)
print("\\nf2 범주:", list(work["f2"].cat.categories))
print("메모리(byte)  원본:", df.memory_usage(deep=True).sum(),
      "→ 변환 후:", work.memory_usage(deep=True).sum())
"""),
    md(r"""
### 수치 반올림: 올림 · 내림 · 반올림 · 버림

계산 결과를 **정수나 특정 자리로 정리**하는 일은 전처리에서 매우 흔합니다. NumPy 함수로 한 번에 처리합니다.

- `np.floor`(내림) · `np.ceil`(올림) · `np.round`(반올림) · `np.trunc`(0 방향 버림)
- 음수에서 `floor`와 `trunc`가 갈립니다: `floor(-1.5) = -2` 지만 `trunc(-1.5) = -1`
"""),
    code("""
sample = df["f5"].head(5)
display(pd.DataFrame({
    "원본": sample,
    "floor": np.floor(sample),
    "ceil": np.ceil(sample),
    "round": np.round(sample),
    "trunc": np.trunc(sample),
}))
"""),
    md(r"""
## 2) 결측값 처리

0–1절에서 데이터를 적재하고 자료형을 정리했습니다. 여기 §2부터 §4까지는 사실 **데이터를 깨끗하게 다듬는 한 묶음**입니다 — 빈 칸을 메우고(결측), 튀는 값을 다스리고(이상치), 변수의 크기를 맞춥니다(스케일). 셋은 보통 이 순서로 이어집니다.

먼저 결측입니다. 결측 대치는 "무엇으로 채우느냐"가 핵심입니다.

| 전략 | 언제 | 코드 |
|------|------|------|
| 전체 통계량 | 결측이 무작위일 때 | `s.fillna(s.median())` |
| **그룹별 통계량** | 그룹마다 분포가 다를 때 | `groupby(key)[col].transform(...)` |
| 최빈값 | 범주형 | `s.fillna(s.mode()[0])` |
| 행 삭제 | 결측이 적고 제거해도 무방할 때 | `dropna(subset=...)` |

`transform`은 그룹별로 계산한 값을 **원래 행 위치에 그대로 되돌려** 주므로 대치에 적합합니다.
`f1`은 `city`마다 수준이 다를 수 있으니 **city별 중앙값**으로 채워 봅니다.
"""),
    code("""
m = df.copy()
print("대치 전 결측 — f1:", m["f1"].isna().sum(), "| f3:", m["f3"].isna().sum())

# (1) 전체 중앙값
overall_median = m["f1"].median()
# (2) city별 중앙값으로 대치 (그룹별 transform)
m["f1"] = m.groupby("city")["f1"].transform(lambda s: s.fillna(s.median()))
# (3) 범주형 f3는 최빈값으로 대치
m["f3"] = m["f3"].fillna(m["f3"].mode()[0])

print("전체 중앙값:", round(overall_median, 2))
print("대치 후 결측 — f1:", m["f1"].isna().sum(), "| f3:", m["f3"].isna().sum())
print("f3 최빈값:", df["f3"].mode()[0])
"""),
    md(r"""
**미션 1.** `f1`의 결측을 *city별 중앙값*으로 채운 뒤, `f1` 전체의 **표준편차**를 소수 셋째 자리까지 구하라.
"""),
    code("""
ans = m["f1"].std()
print(round(ans, 3))
"""),
    md(r"""
## 3) 이상치(outlier) 탐지

빈 칸을 메웠으니, 이제 **비정상적으로 크거나 작은 값**으로 눈을 돌립니다. 이상치는 평균·표준편차 같은 통계와 이후 모델 학습을 모두 흔들 수 있어, 정제 단계에서 함께 다룹니다.

대표적인 두 가지 규칙을 비교합니다. 데이터(`basic2`)의 `Sales`처럼 한쪽으로 크게 치우친 변수에서 결과 차이가 큽니다.

- **IQR 규칙**: $Q_1 - 1.5\,\text{IQR}$ 미만 또는 $Q_3 + 1.5\,\text{IQR}$ 초과를 이상치로 본다. ($\text{IQR}=Q_3-Q_1$)
  분포 가정이 없어 치우친 데이터에 강건합니다.
- **z-score 규칙**: $|z| > 3$ 을 이상치로 본다. ($z=(x-\mu)/\sigma$) 정규분포에 가까울 때 적절합니다.
"""),
    code("""
b2 = pd.read_csv(DE + "basic2.csv")
s = b2["Sales"]

q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
iqr_mask = (s < low) | (s > high)

z = (s - s.mean()) / s.std()
z_mask = z.abs() > 3

print(f"Q1={q1:,.0f}  Q3={q3:,.0f}  IQR={iqr:,.0f}")
print(f"정상 범위: [{low:,.0f}, {high:,.0f}]")
print("IQR 기준 이상치 개수:", int(iqr_mask.sum()))
print("z>3 기준 이상치 개수:", int(z_mask.sum()))

# 처리 예시: IQR 경계로 클리핑(winsorize)
b2["Sales_clipped"] = s.clip(low, high)
print("클리핑 후 최댓값:", f"{b2['Sales_clipped'].max():,.0f}")
"""),
    md(r"""
## 4) 스케일링과 인코딩 — 변수 준비

결측과 이상치를 정리했다면, 마지막은 변수를 **모델이 쓸 수 있는 형태로 준비**하는 단계입니다: 수치형은 크기를 맞추고(스케일링), 범주형(문자)은 숫자로 바꿉니다(인코딩). 먼저 스케일링입니다 — 단위가 제각각인 변수를 그대로 두면 큰 숫자가 작은 숫자를 압도하기 때문입니다.

거리·경사하강 기반 모델(회귀·SVM·신경망 등)은 변수의 **스케일에 민감**합니다.
실습3에서는 공식을 직접 썼는데, 여기서는 같은 일을 `scikit-learn` 변환기로 합니다.

| 변환기 | 공식 | 특징 |
|--------|------|------|
| `StandardScaler` | $(x-\mu)/\sigma$ | 평균 0, 표준편차 1 |
| `MinMaxScaler` | $(x-\min)/(\max-\min)$ | 0~1 범위 |
| `RobustScaler` | $(x-\text{med})/\text{IQR}$ | 중앙값·IQR 사용 → 이상치에 강건 |

> **주의**: 스케일러는 train에 `fit`하고 test에는 `transform`만 합니다. test 통계로 fit하면 정보 누수가 됩니다.
> (이 원칙은 실습5-2 모델링에서 다시 다룹니다.)
"""),
    code("""
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

num = m[["age", "f1", "f5"]].astype("float64")

scaled = pd.DataFrame({
    "f5_standard": StandardScaler().fit_transform(num[["f5"]]).ravel(),
    "f5_minmax":   MinMaxScaler().fit_transform(num[["f5"]]).ravel(),
    "f5_robust":   RobustScaler().fit_transform(num[["f5"]]).ravel(),
})
display(scaled.describe().T[["mean", "std", "min", "max"]])
"""),
    md(r"""
**미션 2.** `f5`를 `MinMaxScaler`로 0~1 변환했을 때, **0.5를 초과**하는 행의 개수를 구하라.
"""),
    code("""
f5_mm = MinMaxScaler().fit_transform(m[["f5"]]).ravel()
print(int((f5_mm > 0.5).sum()))
"""),
    md(r"""
### 범주형 인코딩

스케일링은 수치형을 다뤘습니다. 그런데 모델은 **숫자만** 받기 때문에, 문자로 된 범주형은 숫자로 바꿔야 합니다(인코딩). 두 방식이 기본입니다.

- **라벨 인코딩**: 각 범주에 정수를 부여합니다(`map`/`replace`, `astype("category").cat.codes`, 또는 `LabelEncoder`). 순서가 있는 범주(저·중·고)에 적합.
- **원-핫 인코딩**: 범주마다 0/1 열을 만듭니다(`get_dummies`). 순서 없는 범주에 적합하며, **실습5-2 모델링에서 이 방식**을 씁니다.
"""),
    code("""
from sklearn.preprocessing import LabelEncoder

enc = df.copy()
# 라벨 인코딩: 범주 → 정수 (세 가지 방법은 같은 개념)
enc["city_codes"] = enc["city"].astype("category").cat.codes      # 범주를 정수 코드로 (cat.codes)
enc["f4_label"] = LabelEncoder().fit_transform(enc["f4"])         # sklearn LabelEncoder로 정수화
enc["f2_recode"] = enc["f2"].map({0: "low", 1: "mid", 2: "high"}) # map으로 값 재코딩

# 원-핫 인코딩: 범주마다 0/1 열
onehot = pd.get_dummies(enc["city"], prefix="city")

display(enc[["city", "city_codes", "f4", "f4_label", "f2", "f2_recode"]].head(3))
print("원-핫 결과 열:", list(onehot.columns))
"""),
    md(r"""
## 5) 조건 필터링·선택·정렬

데이터를 깨끗이 정제했으니, 이제 **필요한 부분만 골라내고 정렬**해 원하는 답을 뽑아냅니다. 여기서부터는 정제된 데이터를 '질문에 답하는' 도구로 씁니다.

여러 조건을 조합해 부분집합을 뽑고, 원하는 칸을 골라 값을 바꾸고, 정렬하는 연습입니다. `Titanic`을 사용합니다.

- 불리언 인덱싱: `df[(A) & (B)]` — 각 조건은 괄호로 감싸고 `&`(and)/`|`(or)를 씁니다.
- `query("Pclass == 1 and Gender == 'female'")` — 가독성이 좋습니다.
- `isin`, `between` — 다중 값/범위 조건.
- `sort_values`, `nlargest(n, col)` — 상위 n개.
"""),
    code("""
tt = pd.read_csv(DE + "Titanic.csv")

# 1등석 여성 승객
p1f = tt[(tt["Pclass"] == 1) & (tt["Gender"] == "female")]
print("1등석 여성 수:", len(p1f), "| 생존율:", round(p1f["Survived"].mean(), 3))

# query로 범위(between)·목록(in) 조건 걸기
mid_fare = tt.query("Fare.between(30, 100) and Embarked in ['S', 'C']")
print("운임 30~100 & 승선지 S/C:", len(mid_fare))

# 운임 상위 5명
display(tt.nlargest(5, "Fare")[["Name", "Pclass", "Gender", "Age", "Fare", "Survived"]])
"""),
    md(r"""
### `loc` / `iloc`: 선택과 값 할당

데이터를 고르고 고친다 — 가장 자주 쓰는 두 가지입니다.

- `loc[행조건, 열이름]`: **라벨/조건**으로 선택. 같은 문법으로 **값 할당**도 합니다 — `df.loc[조건, "열"] = 값`
- `iloc[행번호, 열번호]`: **정수 위치**로 선택(정렬 후 상위 k행 등).

`SettingWithCopyWarning`을 피하려면 `df["a"][조건] = ...` 같은 연쇄 인덱싱 대신 **항상 `loc`로 한 번에** 할당합니다.
"""),
    code("""
sel = tt.copy()

# loc: 조건으로 행을 고르고 한 열만 선택
print("60세 이상 평균 운임:", round(sel.loc[sel["Age"] >= 60, "Fare"].mean(), 2))

# loc 조건부 값 할당: 새 파생열을 조건에 따라 채움
sel.loc[sel["Age"] >= 60, "age_band"] = "senior"
sel.loc[sel["Age"] < 60, "age_band"] = "adult"
print("senior 수:", int((sel["age_band"] == "senior").sum()))

# iloc: 운임 내림차순 정렬 후 '위치'로 상위 3행
top3 = sel.sort_values("Fare", ascending=False).iloc[:3]
display(top3[["Name", "Pclass", "Fare"]])
"""),
    md(r"""
**미션 3.** `Pclass == 3` 이면서 `Age >= 30` 인 승객의 **평균 운임(`Fare`)** 을 소수 둘째 자리까지 구하라.
(`Age` 결측 행은 자동으로 조건에서 제외됩니다.)
"""),
    code("""
sub = tt[(tt["Pclass"] == 3) & (tt["Age"] >= 30)]
print(round(sub["Fare"].mean(), 2))
"""),
    md(r"""
## 6) 구간화(binning)와 순위(rank)

연속형을 구간으로 묶으면 비선형 관계를 표로 드러낼 수 있습니다.

- `pd.cut(x, bins=[...])`: **경계값을 직접 지정**(동일 폭/의미 구간).
- `pd.qcut(x, q=4)`: **분위수 기준**으로 같은 개수씩 묶음.
- `rank(method="...")`: 순위 부여(`average`, `min`, `dense` 등).

아래에서 `Fare`를 의미 구간으로 나눠 구간별 생존율을 보면, 운임이 높을수록 생존율이 오르는 경향이 보입니다.
"""),
    code("""
tb = tt.copy()
# 의미 구간으로 운임 구간화
tb["fare_bin"] = pd.cut(tb["Fare"], bins=[0, 10, 30, 100, 1000], right=False,
                        labels=["~10", "10~30", "30~100", "100+"])
print("[운임 구간별 생존율]")
display(tb.groupby("fare_bin", observed=True)["Survived"].agg(["count", "mean"]).round(3))

# 나이를 분위수 4구간으로
tb["age_q"] = pd.qcut(tb["Age"], q=4)
print("[나이 4분위 구간별 인원]")
display(tb["age_q"].value_counts().sort_index())
"""),
    md(r"""
**미션 4.** `Fare`를 `qcut`으로 4분위 구간으로 나눴을 때, **가장 높은 운임 구간**(상위 25%)에 속한 승객의
**생존율**을 소수 셋째 자리까지 구하라.
"""),
    code("""
tt["fare_q"] = pd.qcut(tt["Fare"], 4, labels=[1, 2, 3, 4])
top_fare = tt[tt["fare_q"] == 4]
print(round(top_fare["Survived"].mean(), 3))
"""),
    md(r"""
## 7) 문자열·날짜 처리

### 문자열: `.str` 접근자
`str.contains`, `str.split`, `str.replace`, `str.len` 등으로 텍스트 열을 가공합니다.
`Titanic`의 `Name`에는 호칭(Mr/Mrs/Miss…)이 들어 있어 간단한 파생변수를 만들 수 있습니다.

### 날짜: `to_datetime` + `.dt`
문자열/정수로 들어온 날짜를 `datetime`으로 바꾸면 `.dt.year`, `.dt.weekday` 등 성분 추출이 가능합니다.
`basic2`의 `Date`로 **주말 여부**를 만들어 봅니다. (`weekday`: 월=0 … 토=5, 일=6)
"""),
    code("""
# 문자열: 호칭 포함 여부
print("이름에 'Mrs' 포함:", int(tt["Name"].str.contains("Mrs").sum()))
print("이름에 'Miss' 포함:", int(tt["Name"].str.contains("Miss").sum()))

# 날짜: 주말 파생
b2d = pd.read_csv(DE + "basic2.csv")
b2d["Date"] = pd.to_datetime(b2d["Date"])
b2d["weekday"] = b2d["Date"].dt.weekday
b2d["is_weekend"] = b2d["weekday"] >= 5
print("\\n[주중/주말 Sales 평균]")
display(b2d.groupby("is_weekend")["Sales"].mean().round(0))
"""),
    md(r"""
**미션 5.** `basic2`에서 **주말(토·일)** 의 `Sales` 합계를 정수로 구하라.
"""),
    code("""
print(int(b2d.loc[b2d["is_weekend"], "Sales"].sum()))
"""),
    md(r"""
### 누적합(cumsum)

시간순으로 정렬한 값을 차곡차곡 더해 가는 연산입니다. "누적 매출이 처음으로 전체의 절반을 넘는 날"처럼
**임계점을 찾는 문제**에 자주 쓰입니다. 반드시 **정렬 후** 누적해야 의미가 있습니다.
"""),
    code("""
cs = b2d.sort_values("Date").copy()
cs["cum_sales"] = cs["Sales"].cumsum()
half = cs["Sales"].sum() * 0.5
first_day = cs.loc[cs["cum_sales"] >= half, "Date"].iloc[0]
print("누적 매출이 전체의 50%를 처음 넘는 날:", first_day.date())
display(cs[["Date", "Sales", "cum_sales"]].head())
"""),
    md(r"""
### 연·월 단위 집계

날짜에서 연도·월을 뽑아 `groupby(["year", "month"])`로 묶으면 월별 추세를 집계할 수 있습니다.
"가장 큰 달", "두 번째로 큰 달"처럼 **정렬 후 특정 순위의 값**을 묻는 경우가 많습니다.
"""),
    code("""
ym = b2d.copy()
ym["year"] = ym["Date"].dt.year
ym["month"] = ym["Date"].dt.month

monthly = ym.groupby(["year", "month"])["Sales"].sum().sort_values(ascending=False)
print("[연·월별 매출 합계 — 상위 3]")
display(monthly.head(3))
print("가장 큰 달 매출:", int(monthly.iloc[0]))
print("두 번째로 큰 달 매출:", int(monthly.iloc[1]))   # 정렬 후 .iloc[1]
"""),
    md(r"""
## 8) 상관관계

두 수치형 변수가 **함께 움직이는 정도**를 나타내는 값이 상관계수입니다. `df.corr()`로 한 번에 계산합니다.

- 값의 범위는 -1 ~ 1. 절댓값이 클수록 강한 (선형) 관계입니다.
- 부호는 방향을 뜻합니다(같이 커지면 +, 반대로 움직이면 -).
- 자기 자신과의 상관(대각선=1)은 빼고 봅니다.
"""),
    code("""
num = tt[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]]
corr = num.corr()
print("[상관계수 행렬]")
display(corr.round(3))

# Survived와의 상관을 절댓값 기준으로 정렬
to_survived = corr["Survived"].drop("Survived")
print("[Survived와의 상관계수]")
display(to_survived.sort_values(key=abs, ascending=False).round(3))
"""),
    md(r"""
**미션 6.** 위 변수들 중 `Survived`와 **상관계수의 절댓값이 가장 큰 변수**의 이름과 그 상관계수(소수 셋째)를 구하라.
"""),
    code("""
strongest = to_survived.abs().idxmax()
print(strongest, round(to_survived[strongest], 3))
"""),
    md(r"""
## 9) 종합문제

실제 데이터 처리는 함수 하나로 끝나지 않고 **여러 단계를 연결**하는 경우가 많습니다.
아래 두 문제는 *결측 처리 → 그룹 집계 → 조건/선택 → 집계*의 흐름을 한 번에 묶은 형태입니다.

**종합문제 1.** `Titanic`에서
1. `Age` 결측을 **`Pclass`별 중앙값**으로, `Embarked` 결측을 **최빈값**으로 채운다.
2. 처리 후 `Pclass == 3` 이고 `Age < 20` 인 승객의 **생존율**을 소수 셋째 자리로 구하라.
"""),
    code("""
cap = pd.read_csv(DE + "Titanic.csv")
# 1) Pclass별 중앙값으로 Age 대치
cap["Age"] = cap.groupby("Pclass")["Age"].transform(lambda s: s.fillna(s.median()))
# 2) Embarked 최빈값 대치
cap["Embarked"] = cap["Embarked"].fillna(cap["Embarked"].mode()[0])
# 3) 조건부 생존율
young3 = cap[(cap["Pclass"] == 3) & (cap["Age"] < 20)]
print("대상 인원:", len(young3))
print("생존율:", round(young3["Survived"].mean(), 3))
"""),
    md(r"""
**종합문제 2.** (그룹별 최댓값 선택형) `Titanic`에서 `Embarked` 결측을 최빈값으로 채운 뒤,
**각 `Pclass`에서 생존율이 가장 높은 승선항구(`Embarked`)** 를 찾고,
그 (`Pclass`, `Embarked`) 그룹들에 속한 **승객 수의 합**을 구하라.

> 흐름: `groupby([Pclass, Embarked])`로 생존율 → 각 `Pclass`별 `idxmax`로 최고 그룹 선택 → 해당 그룹들의 인원 합산.
"""),
    code("""
g = pd.read_csv(DE + "Titanic.csv")
g["Embarked"] = g["Embarked"].fillna(g["Embarked"].mode()[0])

rate = g.groupby(["Pclass", "Embarked"])["Survived"].mean()
best = rate.groupby("Pclass").idxmax()          # 각 Pclass에서 생존율 최고 (Pclass, Embarked)
sizes = g.groupby(["Pclass", "Embarked"]).size()

print("Pclass별 최고 생존율 항구:", list(best))
print("해당 그룹들의 승객 수 합:", int(sum(sizes[k] for k in best)))
"""),
    md(r"""
## 생각해보기

1. **미션 1**에서 그룹별 중앙값 대신 *전체 평균*으로 채우면 표준편차는 어떻게 달라지나요? 왜 그럴까요?
2. `Sales`의 이상치를 IQR로 **제거**했을 때와 경계로 **클리핑**했을 때, 평균은 어떻게 달라지나요?
3. `RobustScaler`가 이상치가 많은 변수에서 `StandardScaler`보다 안정적인 이유를 공식으로 설명해 보세요.
4. **미션 4**를 `cut`(동일 폭)으로 바꾸면 구간별 인원이 크게 불균형해집니다. 이유는 무엇일까요?
5. **종합문제 1**에서 `Age`를 `Pclass`별이 아니라 `Pclass × Gender`별 중앙값으로 대치하면 답이 달라지나요?
6. **종합문제 2**를 `idxmax` 없이 `sort_values` + `groupby().head(1)`로도 풀 수 있습니다. 두 방법의 결과가 같은지 확인해 보세요.
"""),
]

write_nb("practices/practice4_1.ipynb", cells)
