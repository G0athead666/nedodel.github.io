# main.py
import time
import os # Добавим os для отладочных путей

print(f"DEBUG APP: Start of main.py execution at {time.strftime('%H:%M:%S')}")

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
# Убедитесь, что model_loader.py находится в той же директории или импортируется корректно
from model_loader import predict_iris

print(f"DEBUG APP: FastAPI and model_loader imported at {time.strftime('%H:%M:%S')}")

app = FastAPI()
print(f"DEBUG APP: FastAPI app initialized at {time.strftime('%H:%M:%S')}")

# Убедитесь, что директория 'templates' существует относительно текущей рабочей директории
# (которая, по идее, должна быть /content/sample_data/iris/app/)
# Можно добавить проверку:
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
print(f"DEBUG APP: Templates directory set to: {templates_dir} at {time.strftime('%H:%M:%S')}")
templates = Jinja2Templates(directory=templates_dir) # Используем абсолютный путь для Jinja2Templates

@app.get("/", response_class=HTMLResponse)
async def form(request: Request):
    print(f"DEBUG APP: '/ (GET)' endpoint called at {time.strftime('%H:%M:%S')}")
    return templates.TemplateResponse("form.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    sepal_length: float = Form(...),
    sepal_width: float = Form(...),
    petal_length: float = Form(...),
    petal_width: float = Form(...)
):
    print(f"DEBUG APP: '/predict (POST)' endpoint called at {time.strftime('%H:%M:%S')}")
    features = [sepal_length, sepal_width, petal_length, petal_width]
    print(f"DEBUG APP: Features for prediction: {features} at {time.strftime('%H:%M:%S')}")
    result = predict_iris(features)
    print(f"DEBUG APP: Prediction result: {result} at {time.strftime('%H:%M:%S')}")

    return templates.TemplateResponse("form.html",
                                     {"request": request,
                                      "result": result})

print(f"DEBUG APP: End of main.py script execution (before Uvicorn starts serving) at {time.strftime('%H:%M:%S')}")
