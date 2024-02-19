from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import train, predict
from typing import List, Dict
import json

app = FastAPI()

class TrainData(BaseModel):
    category: str
    price: float
    promotion: int
    discount_perc: int
    channel: str
    purchased: int

class TrainRequest(BaseModel):
    save_path: str
    data: List[TrainData]

class TrainResponse(BaseModel):
    accuracy: float
    status: str

class PredictRequest(BaseModel):
    category: str
    price: float
    promotion: int
    discount_perc: int
    channel: str

class PredictResponse(BaseModel):
    result: str
    status: str

@app.get("/")
async def root():
    return {"message": "Service is up and running."}

@app.post("/train", response_model=TrainResponse)
async def train_model(train_request: TrainRequest):
    try:
        with open("data_train.json", "r") as file:
            existing_data = json.load(file)
        data = existing_data + [item.dict() for item in train_request.data]
        accuracy = train(data, train_request.save_path)
        return {"accuracy": accuracy, "status": "success"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictResponse)
async def predict_model(predict_request: PredictRequest):
    try:
        result = predict(predict_request.dict(), "model_learning")
        return {"result": result, "status": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))