from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import numpy as np
import uvicorn
from sklearn.ensemble import RandomForestRegressor

app = FastAPI()
templates = Jinja2Templates(directory="templates")

# Train model at module load
np.random.seed(42)
n = 1000
study = np.random.uniform(1, 12, n)
attendance = np.random.uniform(60, 100, n)
prev = np.random.uniform(40, 100, n)
extra = np.random.randint(0, 2, n)
parent_edu = np.random.randint(1, 4, n)
score = (study*4 + attendance*0.4 + prev*0.3 + extra*5 + parent_edu*3 + np.random.normal(0, 5, n))
score = np.clip(score, 0, 100)
X = np.column_stack([study, attendance, prev, extra, parent_edu])
model = RandomForestRegressor(n_estimators=50, random_state=42)
model.fit(X, score)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class StudentData(BaseModel):
    study_hours: float
    attendance: float
    prev_grade: float
    extracurricular: int
    parent_edu: int


@app.post("/predict")
async def predict(data: StudentData):
    x = np.array([[data.study_hours, data.attendance, data.prev_grade, data.extracurricular, data.parent_edu]])
    sc = float(model.predict(x)[0])
    grade = 'A' if sc >= 90 else 'B' if sc >= 75 else 'C' if sc >= 60 else 'D' if sc >= 50 else 'F'
    return {"score": round(sc, 1), "grade": grade}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5006)
