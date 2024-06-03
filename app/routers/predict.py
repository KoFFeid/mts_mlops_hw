import io

import pandas as pd
import matplotlib.pyplot as plt
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import RedirectResponse


from src import scorer


router = APIRouter(
    prefix="/predict",
    tags=["predict"]
)

@router.post("/")
def predict(request: Request, file: UploadFile = File(...)):

    model = request.app.state.model
    predproc = request.app.state.predproc
    
    try:
        df = pd.read_csv(file.file)
        file.file.close()
    except Exception:
        print("Проблема с загрузкой файла")

    id = df.client_id
    predproc.transform_data_frame(df)
    predictions, probability = scorer.make_pred(model, df, id)

    predictions.to_csv("data/predictions.csv", index=False)
    probability.to_csv("data/probability.csv", index=False)

    return {"message": "Make predictions successfully"}
