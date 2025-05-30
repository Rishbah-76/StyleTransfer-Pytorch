# main.py
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import os
from style_transfer import StyleTransferModel
from utils import gram_matrix
from torchvision.transforms import transforms

# Create static directory if it doesn't exist
os.makedirs("static", exist_ok=True)

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")
model = StyleTransferModel()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse)
async def upload(request: Request, content_img: UploadFile = File(...), style_img: UploadFile = File(...)):
    # Save uploaded images
    content = Image.open(io.BytesIO(await content_img.read())).convert("RGB")
    style = Image.open(io.BytesIO(await style_img.read())).convert("RGB")
    
    # Save original images
    content.save("static/content.jpg")
    style.save("static/style.jpg")

    # Process the style transfer
    output = model.run(content, style)
    output_img = transforms.ToPILImage()(output)
    output_img.save("static/output.jpg")

    return templates.TemplateResponse("index.html", {
        "request": request,
        "result": True,
        "content_path": "/static/content.jpg",
        "style_path": "/static/style.jpg",
        "output_path": "/static/output.jpg"
    })
