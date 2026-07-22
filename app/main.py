import os
import re

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

app = FastAPI(title="Student Evaluation Score Predictor")
templates = Jinja2Templates(directory="templates")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "datathon_data", "train.csv")

TARGET_COL = "Degerlendirme Puani"

# ---------------------------------------------------------------------------
# Turkish-aware text helpers
# ---------------------------------------------------------------------------


def tr_lower(text: str) -> str:
    """Lowercase a string the Turkish-correct way (İ -> i, I -> ı)."""
    return text.replace("İ", "i").replace("I", "ı").lower()


def clean_binary(value) -> str:
    """Normalizes free-form Evet/Hayır style answers."""
    if pd.isna(value):
        return "Bilinmiyor"
    s = tr_lower(str(value).strip())
    if "evet" in s:
        return "Evet"
    if "hayir" in s or "hayır" in s:
        return "Hayır"
    return "Bilinmiyor"


def clean_gender(value) -> str:
    if pd.isna(value):
        return "Bilinmiyor"
    s = tr_lower(str(value).strip())
    if "kadin" in s or "kadın" in s:
        return "Kadın"
    if "erkek" in s:
        return "Erkek"
    if "belirtmek" in s:
        return "Belirtmek istemiyorum"
    return "Bilinmiyor"


def clean_university_type(value) -> str:
    if pd.isna(value):
        return "Bilinmiyor"
    s = tr_lower(str(value).strip())
    if "devlet" in s:
        return "Devlet"
    if "ozel" in s or "özel" in s:
        return "Özel"
    return "Bilinmiyor"


def clean_high_school_type(value) -> str:
    if pd.isna(value):
        return "Bilinmiyor"
    s = tr_lower(str(value).strip())
    if "anadolu" in s:
        return "Anadolu Lisesi"
    if "fen" in s:
        return "Fen Lisesi"
    if "meslek" in s:
        return "Meslek Lisesi"
    if "imam hatip" in s:
        return "İmam Hatip Lisesi"
    if "duz" in s or "düz" in s:
        return "Düz Lise"
    if "ozel" in s or "özel" in s:
        return "Özel Lise"
    if "devlet" in s:
        return "Devlet Lisesi"
    return "Diğer"


def clean_education_level(value) -> str:
    if pd.isna(value):
        return "Bilinmiyor"
    s = tr_lower(str(value).strip())
    if s in ("0", ""):
        return "Bilinmiyor"
    if "ilkokul" in s:
        return "İlkokul"
    if "ortaokul" in s:
        return "Ortaokul"
    if "yuksek lisans" in s or "yüksek lisans" in s or "doktora" in s or "doktara" in s:
        return "Yüksek Lisans/Doktora"
    if "universite" in s or "üniversite" in s:
        return "Üniversite"
    if "egitim yok" in s or "eğitim yok" in s:
        return "Eğitim Yok"
    if "lise" in s:
        return "Lise"
    return "Diğer"


CITY_ALIASES = {
    "Istanbul": "İstanbul",
    "Izmir": "İzmir",
    "Diyarbakir": "Diyarbakır",
    "Sanliurfa": "Şanlıurfa",
}


def build_city_feature(raw_series: pd.Series, top_n: int = 20) -> pd.Series:
    normalized = raw_series.astype("object").apply(
        lambda v: np.nan if pd.isna(v) else CITY_ALIASES.get(str(v).strip(), str(v).strip())
    )
    top_cities = normalized.value_counts().head(top_n).index.tolist()

    def bucket(v):
        if pd.isna(v):
            return "Bilinmiyor"
        return v if v in top_cities else "Diğer"

    return normalized.apply(bucket)


def parse_mean_number(value) -> float:
    """Extracts number(s) from a free-text range like '3.00 - 3.50' and averages them."""
    if pd.isna(value):
        return np.nan
    nums = re.findall(r"\d+\.?\d*", str(value))
    if not nums:
        return np.nan
    nums = [float(n) for n in nums]
    return sum(nums) / len(nums)


def parse_grade_100(value) -> float:
    """High-school grade column mixes a 0-100 scale and a 0-4 GPA scale in the same
    column (e.g. '75 - 100' vs '4.00 - 3.50'). Detect the scale from the magnitude
    of the numbers found and normalize everything onto a 0-100 scale."""
    if pd.isna(value):
        return np.nan
    nums = re.findall(r"\d+\.?\d*", str(value))
    if not nums:
        return np.nan
    nums = [float(n) for n in nums]
    mean_val = sum(nums) / len(nums)
    if max(nums) <= 5:  # looks like a 0-4 GPA scale
        return mean_val * 25.0
    return mean_val


def parse_university_year(value) -> float:
    if pd.isna(value):
        return np.nan
    s = tr_lower(str(value).strip())
    if "hazirlik" in s or "hazırlık" in s:
        return 0.0
    if "mezun" in s:
        return 5.0
    if "yuksek lisans" in s or "yüksek lisans" in s or "tez" in s:
        return 6.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def parse_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


# ---------------------------------------------------------------------------
# Feature configuration
# ---------------------------------------------------------------------------

# Ordered list of the model's input features (friendly, snake_case API names).
FEATURE_COLUMNS = [
    "gender",
    "residence_city",
    "university_type",
    "university_year",
    "university_gpa",
    "high_school_type",
    "high_school_grade",
    "has_scholarship",
    "scholarship_percentage",
    "mother_education",
    "father_education",
    "sibling_count",
    "entrepreneurship_club_member",
    "knows_english",
]

CATEGORICAL_FEATURES = [
    "gender",
    "residence_city",
    "university_type",
    "high_school_type",
    "has_scholarship",
    "mother_education",
    "father_education",
    "entrepreneurship_club_member",
    "knows_english",
]

NUMERIC_FEATURES = [c for c in FEATURE_COLUMNS if c not in CATEGORICAL_FEATURES]


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)

    feat["gender"] = df["Cinsiyet"].apply(clean_gender)
    feat["residence_city"] = build_city_feature(df["Ikametgah Sehri"])
    feat["university_type"] = df["Universite Turu"].apply(clean_university_type)
    feat["university_year"] = df["Universite Kacinci Sinif"].apply(parse_university_year)
    feat["university_gpa"] = df["Universite Not Ortalamasi"].apply(parse_mean_number)
    feat["high_school_type"] = df["Lise Turu"].apply(clean_high_school_type)
    feat["high_school_grade"] = df["Lise Mezuniyet Notu"].apply(parse_grade_100)
    feat["has_scholarship"] = df["Burs Aliyor mu?"].apply(clean_binary)
    feat["scholarship_percentage"] = df["Burslu ise Burs Yuzdesi"].apply(parse_float)
    feat["mother_education"] = df["Anne Egitim Durumu"].apply(clean_education_level)
    feat["father_education"] = df["Baba Egitim Durumu"].apply(clean_education_level)
    feat["sibling_count"] = df["Kardes Sayisi"].apply(parse_float)
    feat["entrepreneurship_club_member"] = df[
        "Girisimcilik Kulupleri Tarzi Bir Kulube Uye misiniz?"
    ].apply(clean_binary)
    feat["knows_english"] = df["Ingilizce Biliyor musunuz?"].apply(clean_binary)

    # Missing-value handling.
    # NaN scholarship % almost always means "not a scholarship recipient" -> 0.
    feat["scholarship_percentage"] = feat["scholarship_percentage"].fillna(0).clip(0, 100)
    feat["university_gpa"] = feat["university_gpa"].fillna(feat["university_gpa"].median()).clip(0, 4)
    feat["high_school_grade"] = feat["high_school_grade"].fillna(feat["high_school_grade"].median()).clip(0, 100)
    feat["university_year"] = feat["university_year"].fillna(feat["university_year"].median()).clip(0, 6)
    feat["sibling_count"] = feat["sibling_count"].fillna(feat["sibling_count"].median()).clip(0, 15)

    return feat


# ---------------------------------------------------------------------------
# Train the model once at module load
# ---------------------------------------------------------------------------

_raw_df = pd.read_csv(DATA_PATH, encoding="utf-8", low_memory=False)
_raw_df = _raw_df.dropna(subset=[TARGET_COL])

_features = build_feature_frame(_raw_df)
_target = _raw_df[TARGET_COL].astype(float)

ENCODERS: dict[str, LabelEncoder] = {}
for col in CATEGORICAL_FEATURES:
    le = LabelEncoder()
    _features[col + "_enc"] = le.fit_transform(_features[col].astype(str))
    ENCODERS[col] = le

MODEL_INPUT_COLUMNS = [
    (c + "_enc" if c in CATEGORICAL_FEATURES else c) for c in FEATURE_COLUMNS
]

X = _features[MODEL_INPUT_COLUMNS].values
y = _target.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=18,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

MODEL_R2 = float(r2_score(y_test, model.predict(X_test)))

FEATURE_IMPORTANCE = [
    {"feature": name, "importance": round(float(score), 4)}
    for name, score in sorted(
        zip(FEATURE_COLUMNS, model.feature_importances_.tolist()),
        key=lambda t: t[1],
        reverse=True,
    )
]

OPTIONS = {col: ENCODERS[col].classes_.tolist() for col in CATEGORICAL_FEATURES}


def encode_category(col: str, value: str) -> int:
    le = ENCODERS[col]
    if value in le.classes_:
        return int(le.transform([value])[0])
    if "Bilinmiyor" in le.classes_:
        return int(le.transform(["Bilinmiyor"])[0])
    return 0


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class StudentEvaluationRequest(BaseModel):
    gender: str = Field(..., description="Öğrencinin cinsiyeti", examples=["Kadın"])
    residence_city: str = Field(..., description="İkametgah şehri", examples=["İstanbul"])
    university_type: str = Field(..., description="Üniversite türü", examples=["Devlet"])
    university_year: float = Field(..., ge=0, le=6, description="Üniversite kaçıncı sınıf (0=Hazırlık, 5=Mezun, 6=Yüksek Lisans/Tez)")
    university_gpa: float = Field(..., ge=0, le=4, description="Üniversite not ortalaması (4.0 skalası)")
    high_school_type: str = Field(..., description="Lise türü", examples=["Anadolu Lisesi"])
    high_school_grade: float = Field(..., ge=0, le=100, description="Lise mezuniyet notu (0-100 skalasına normalize edilmiş)")
    has_scholarship: str = Field(..., description="Burs alıyor mu?", examples=["Evet"])
    scholarship_percentage: float = Field(..., ge=0, le=100, description="Burs yüzdesi (burssuzsa 0)")
    mother_education: str = Field(..., description="Anne eğitim durumu", examples=["Lise"])
    father_education: str = Field(..., description="Baba eğitim durumu", examples=["Üniversite"])
    sibling_count: float = Field(..., ge=0, le=15, description="Kardeş sayısı")
    entrepreneurship_club_member: str = Field(..., description="Girişimcilik kulübü üyesi mi?", examples=["Hayır"])
    knows_english: str = Field(..., description="İngilizce biliyor mu?", examples=["Evet"])


class FeatureImportanceItem(BaseModel):
    feature: str
    importance: float


class PredictResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    predicted_score: float
    feature_importance: list[FeatureImportanceItem]
    model_r2: float


@app.get("/options")
async def get_options():
    return OPTIONS


@app.post("/predict", response_model=PredictResponse)
async def predict(data: StudentEvaluationRequest):
    row = []
    for col in FEATURE_COLUMNS:
        value = getattr(data, col)
        if col in CATEGORICAL_FEATURES:
            row.append(encode_category(col, value))
        else:
            row.append(float(value))

    x = np.array([row])
    predicted = float(model.predict(x)[0])
    predicted = max(0.0, round(predicted, 2))

    return PredictResponse(
        predicted_score=predicted,
        feature_importance=[FeatureImportanceItem(**item) for item in FEATURE_IMPORTANCE],
        model_r2=round(MODEL_R2, 4),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5006)
