import io

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse

import pandas as pd
import matplotlib.pyplot as plt

router = APIRouter(
    prefix="/downloads",
    tags=["download"]
)


@router.get("/predictions")
async def predictions_download(request: Request):
    return FileResponse("./data/predictions.csv", media_type="text/csv")


@router.get("/feature_importance")
async def download_feature_importance(request: Request):

    model = request.app.state.model

    importance = pd.Series( 
        model.get_feature_importance(),
        index=model.feature_names_,
    )
    importance.sort_values(ascending=False)[:5].to_json("./data/feature_importance.json", force_ascii=False)
    return FileResponse("./data/feature_importance.json", media_type="json")


@router.get("/predictionss_distrib")
async def download_distrib_plot(request: Request):
    probs = pd.read_csv("./data/probability.csv")
    fig, ax = plt.subplots()
    plt.hist(probs["probability"])
    plt.title("Probability distribution")
    plt.savefig("./data/distrib_plot.png", format="png")
    return FileResponse("./data/distrib_plot.png", media_type="image/png")