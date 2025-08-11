from fastapi import FastAPI, HTTPException
from transformers import AutoModelForCausalLM, AutoTokenizer
from pydantic import BaseModel
import torch
from peft import PeftModel
import uvicorn
import os

# Initialize FastAPI app
app = FastAPI(title="Mistral Model API")

# Model configuration
MODEL_PATH = r"C:\dummy dataser\final_model"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 2048
OFFLOAD_DIR = "offload_dir"  # Directory for model offloading

# Create offload directory if it doesn't exist
os.makedirs(OFFLOAD_DIR, exist_ok=True)

class GenerationRequest(BaseModel):
    prompt: str
    max_length: int = MAX_LENGTH
    temperature: float = 0.7
    top_p: float = 0.9
    num_return_sequences: int = 1

# Load model and tokenizer
@app.on_event("startup")
async def load_model():
    global model, tokenizer
    try:
        print(f"Loading model from {MODEL_PATH}")
        
        # Initialize tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            padding_side="right",
            use_fast=False
        )
        
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load the base model with 4-bit quantization
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_PATH,
            load_in_4bit=True,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder=OFFLOAD_DIR,
            trust_remote_code=True
        )
        
        print("Model loaded successfully")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise RuntimeError(f"Failed to load model: {str(e)}")

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    try:
        # Format the prompt with Mistral's instruction format
        formatted_prompt = f"<s>[INST] {request.prompt} [/INST]"
        
        # Tokenize input
        inputs = tokenizer(
            formatted_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=request.max_length
        ).to(DEVICE)

        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=request.max_length,
                temperature=request.temperature,
                top_p=request.top_p,
                num_return_sequences=request.num_return_sequences,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode and return generated text
        generated_texts = [
            tokenizer.decode(output, skip_special_tokens=True)
            for output in outputs
        ]

        return {"generated_texts": generated_texts}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": DEVICE}

# Run the API server
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    ) 