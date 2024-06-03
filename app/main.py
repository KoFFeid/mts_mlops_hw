from catboost import CatBoostClassifier
from fastapi import FastAPI

from routers import downloads, predict, main_route
from src.preprocessing import Predproc


def create_app() -> FastAPI:
    app = FastAPI()
    app.include_router(downloads.router)
    app.include_router(predict.router)
    app.include_router(main_route.router)
    prepare_model(app)
    return app



def prepare_model(app: FastAPI) -> None:
    model = CatBoostClassifier()
    model.load_model('./model/catboost_mts.cbm')
    load_predproc = Predproc()
    load_predproc.fit('./model/train.csv')
    app.state.model = model
    app.state.predproc = load_predproc


app = create_app()