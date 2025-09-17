import os
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

from .schemas import DetectionResponse, Detection, BoundingBox, ErrorResponse
from .model import inference


app = FastAPI(
    title="T-Bank logo detection",
    description="REST API сервис для обнаружения логотипа Т-Банка на загружаемом изображении и возвращении координат найденных логотипов"
)

ALLOWED_FORMATS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@app.get("/")
def root():
    return {
        "status": "ok",
        "docs": "/docs",
        "message": "T-Bank logo detection API is running"
    }


@app.post(
    "/detect", 
    response_model=DetectionResponse,
    responses={400: {"model": ErrorResponse}}
)
async def detect_logo(file: UploadFile = File(...)):
    """
    Детекция логотипа Т-банка на изображении

    Args:
        file: Загружаемое изображение (JPG, JPEG, PNG, BMP, WEBP)

    Returns:
        DetectionResponse: Результаты детекции с координатами найденных логотипов
    """
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_FORMATS:
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error="UnsupportedFileExtension",
                detail=f"Недопустимое расширение файла: {ext}. Разрешены: {', '.join(ALLOWED_FORMATS)}"
            ).dict()
        )
    
    contents = await file.read()
    result = inference(contents)

    if isinstance(result, ErrorResponse):
        return JSONResponse(
            status_code=400,
            content=result.dict()
        )

    return result


