import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Load the fine-tuned model and tokenizer
model_path = "./final_model"

print("Loading model and tokenizer...")

# Apply 4-bit or 8-bit quantization to reduce VRAM usage
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Set to True for 4-bit quantization, use `load_in_8bit=True` for 8-bit
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for best performance
    bnb_4bit_use_double_quant=True,  # Improves memory efficiency
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model with quantization & offloading
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,  # Apply quantization
    device_map="auto",  # Automatically assigns layers to GPU/CPU
    offload_folder="offload_dir",  # Saves offloaded layers to disk if needed
)

def generate_response(prompt, max_length=1024):  # Increase max_length
    print(f"\nInput: {prompt}")

    # Tokenize the input
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # Generate response
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_length=max_length,  # Increase max tokens
            temperature=0.7,  # Adjust for creativity
            top_p=0.9,  # Sampling diversity
            repetition_penalty=1.1,  # Avoid repetition
            eos_token_id=tokenizer.eos_token_id,  # Ensure proper stopping
            do_sample=True,  # Enable sampling for more natural responses
        )

    # Decode and print the output
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"\nResponse: {response}")
    return response


# Example prompt for testing
test_prompt = """ Given the following financial profile of a user, assess their risk tolerance, 
suggest an investment portfolio allocation, and provide adjustments based on their preferences.

User's Financial Profile:

Primary source of income: Salary
Income stability: Somewhat stable
Monthly savings/investments: Less than 10%
Outstanding loans/EMIs: No
Percentage of income spent on loans: More than 50%
Credit card usage: Rarely
Preferred investment options: None
Reaction to a 20% investment drop: Watch
Emergency fund: No
Missed a payment in the last 12 months: Yes """
generate_response(test_prompt)
