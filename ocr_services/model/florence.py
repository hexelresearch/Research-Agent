import torch
from transformers import AutoProcessor, AutoModelForCausalLM
import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cache for model and processor
model_cache = None
processor_cache = None

async def get_florence():
    """
    Load the Florence model and processor, using cached versions if available.
    
    Returns:
        model: Loaded Florence model.
        processor: Loaded processor for the Florence model.
    """
    global model_cache, processor_cache
    try:
        # Return cached model and processor if available
        if model_cache and processor_cache:
            logger.info("Using cached model and processor.")
            return model_cache, processor_cache
        
        # Otherwise, load model and processor in a separate thread
        device, torch_dtype = await device_dtype()  # Await device setup
        logger.info("Loading model and processor...")
        
        model = await asyncio.to_thread(
            AutoModelForCausalLM.from_pretrained,
            "microsoft/Florence-2-large-ft",
            torch_dtype=torch_dtype,
            trust_remote_code=True
        )
        model = model.to(device)
        
        processor = await asyncio.to_thread(
            AutoProcessor.from_pretrained,
            "microsoft/Florence-2-large-ft",
            trust_remote_code=True
        )
        
        # Cache the model and processor
        model_cache = model
        processor_cache = processor
        logger.info("Model and processor loaded successfully.")
        
        return model, processor
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise

async def device_dtype():
    """
    Setup device and dtype for model usage.
    
    Returns:
        device: The device to run the model on (cuda or cpu).
        torch_dtype: The data type for the model (float16 or float32).
    """
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return device, torch_dtype
    except Exception as e:
        logger.error(f"Failed to setup device and dtype: {e}")
        raise
