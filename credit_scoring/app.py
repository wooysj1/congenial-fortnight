from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from model import load_model

app = FastAPI()
model = None


class CreditFeatures(BaseModel):
    # Example fields - extend as needed
    income: float
    age: int
    loan_amount: float
    existing_loans: int


@app.on_event("startup")
def startup_event():
    global model
    model = load_model('credit_model.pkl')


@app.post('/score')
def score(features: CreditFeatures):
    df = pd.DataFrame([features.dict()])
    prob = model.predict_proba(df)[:, 1][0]
    return {"credit_score": prob}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
