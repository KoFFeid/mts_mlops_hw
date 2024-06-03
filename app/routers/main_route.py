import io

import pandas as pd
import matplotlib.pyplot as plt
from fastapi import APIRouter, File, UploadFile, Request
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse

from src import scorer


router = APIRouter()

@router.get("/")
async def main():
    content = """
    <body>
    <form action="/predict" method="POST" enctype="multipart/form-data">
        <label for="file">Upload a file for prediction:</label>
        <input type="file" id="file" name="file"><br />
        <button>Predict</button>
    </form>

    <form action="/downloads/predictions" method="GET" enctype="multipart/form-data">
        <button>Download predictions</button>
    </form>

    <form action="/downloads/feature_importance" method="GET" enctype="multipart/form-data">
        <button>Get importances</button>
    </form>

    <form action="/downloads/predictionss_distrib" method="GET" enctype="multipart/form-data">
        <button>Get scores distribution</button>
    </form>
    </body>
    """
    return HTMLResponse(content=content)
