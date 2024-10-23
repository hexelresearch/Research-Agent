import cv2
import logging
from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from io import BytesIO
from typing import List
from model.florence import get_florence, device_dtype
from PIL import Image
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Define the response model
class OCRResponse(BaseModel):
    filename: str
    result: str

# Initialize model and processor asynchronously
async def initialize_model_and_processor():
    logger.info("Initializing model and processor.")
    model, processor = await get_florence()  # Await the async function
    device, torch_dtype = await device_dtype()  # Await the async function
    return model, processor, device, torch_dtype

# Process images and generate OCR results
async def ocr_images(files: List[UploadFile]):
    logger.info("Starting OCR processing for uploaded files.")
    
    # Initialize model, processor, device, and dtype
    model, processor, device, torch_dtype = await initialize_model_and_processor()

    data = []

    for file in files:
        logger.info(f"Processing file: {file.filename}")
        try:
            # Read image file using OpenCV
            contents = await file.read()
            image_array = np.frombuffer(contents, np.uint8)
            image_cv = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Convert BGR to RGB format
            image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)

            # Convert the image to PIL format for the processor
            image = Image.fromarray(image_rgb)

            # Preprocess input
            inputs = processor(text="<OCR>", images=image, return_tensors="pt").to(device, torch_dtype)

            # Generate captions
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=1024,
                do_sample=False,
                num_beams=3
            )

            # Decode the generated text
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]

            # Post-process the generated text
            parsed_answer = processor.post_process_generation(generated_text, task="<OCR>", image_size=(image.width, image.height))

            # Extract the relevant text from the parsed answer
            result_string = parsed_answer.get("<OCR>", "")  # Adjust the key based on actual output structure

            # Append result to data
            data.append(OCRResponse(filename=file.filename, result=result_string))

            logger.info(f"Successfully processed file: {file.filename}")

        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}: {str(e)}")

    logger.info("OCR processing completed.")
    return data

