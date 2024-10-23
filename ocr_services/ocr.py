from fastapi import FastAPI, UploadFile, File, HTTPException
from services.images_ocr import ocr_images
from services.pdfs_ocr import ocr_pdf
from typing import List
import logging

# Define the API router
app = FastAPI()

# Configure logger
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

@app.post("/ocr_images")
async def ocr_images_endpoint(files: List[UploadFile] = File(...)):
    logger.info("Received request to perform OCR on images.")
    try:
        data = await ocr_images(files)
        logger.info("Data extracted from images successfully.")
        return {"message": "Data has been extracted from images.", "data": data}
    except Exception as e:
        logger.error("Error extracting data from images: %s", str(e))
        raise HTTPException(status_code=500, detail="Error extracting data from images.")

@app.post("/ocr_pdfs")
async def ocr_pdfs_endpoint(files: List[UploadFile] = File(...)):
    logger.info("Received request to perform OCR on PDFs.")
    try:
        data = await ocr_pdf(files)
        logger.info("Data extracted from PDFs successfully.")
        return {"message": "Data has been extracted from PDFs.", "data": data}
    except Exception as e:
        logger.error("Error extracting data from PDFs: %s", str(e))
        raise HTTPException(status_code=500, detail="Error extracting data from PDFs.")
