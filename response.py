import pickle
from typing import Literal
from pydantic import BaseModel, Field
from typing import Dict, Any
from pydantic import ConfigDict
from fastapi import FastAPI
import uvicorn

#request
class Customer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: int
    education: Literal["graduation", "master", "phd"]
    marital_status: Literal["single", "together", "married", "divorced"]
    income: float = Field(..., ge=0)
    kidhome: int = Field(..., ge=0, le=2)
    teenhome: int = Field(..., ge=0, le=2)
    recency: int = Field(..., ge=0, le=99)

    mntwines: float = Field(..., ge=0)
    mntfruits: float = Field(..., ge=0)
    mntmeatproducts: float = Field(..., ge=0)
    mntfishproducts: float = Field(..., ge=0)
    mntsweetproducts: float = Field(..., ge=0)
    mntgoldprods: float = Field(..., ge=0)

    numdealspurchases: int = Field(..., ge=0)
    numwebpurchases: int = Field(..., ge=0)
    numcatalogpurchases: int = Field(..., ge=0)
    numstorepurchases: int = Field(..., ge=0)
    numwebvisitsmonth: int = Field(..., ge=0)

    acceptedcmp3: int = Field(..., ge=0, le=1)
    acceptedcmp4: int = Field(..., ge=0, le=1)
    acceptedcmp5: int = Field(..., ge=0, le=1)
    acceptedcmp1: int = Field(..., ge=0, le=1)
    acceptedcmp2: int = Field(..., ge=0, le=1)

    complain: int = Field(..., ge=0, le=1)

    age: int = Field(..., ge=18)  # customer data minimum ~28
    customer_days: int = Field(..., ge=0)
    total_purchases: int = Field(..., ge=0)
    total_spending: float = Field(..., ge=0)

    previous_response_rate: float = Field(..., ge=0, le=1)

#response
class PredictResponse(BaseModel):
    response_probability: float
    will_respond: bool

app = FastAPI(title="Marketing Campaign Response Predictor")
with open ('model.bin', 'rb') as f_in:
    pipeline = pickle.load(f_in)


def predict_single(customer):
    result = pipeline.predict_proba([customer])[0, 1]
    return float(result)

@app.post('/response')
def predict(customer: Customer) -> PredictResponse:
    prob = predict_single(customer.model_dump())
 
    return PredictResponse(
        response_probability=prob,
        will_respond = prob >= 0.5
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)