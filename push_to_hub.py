from huggingface_hub import HfApi, whoami
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub.utils import validate_repo_id, HfHubHTTPError
import torch
import gc
import os
import time

def free_memory():
    """Aggressively free memory"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        time.sleep(1)

def push_to_hub(local_model_path, repo_name, organization=None):
    """
    Push a local model to the Hugging Face Hub using 4-bit quantization on GPU only
    """
    try:
        # Memory management settings
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:128'
        
        # Verify token
        username = verify_token()
        api = HfApi()
        
        # Set repository name
        full_repo_name = f"{organization}/{repo_name}" if organization else f"{username}/{repo_name}"
        validate_repo_id(full_repo_name)
        
        print(f"Creating repository: {full_repo_name}")
        api.create_repo(
            repo_id=full_repo_name,
            private=True,
            exist_ok=True
        )
        
        # First push tokenizer
        print("Loading and pushing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(local_model_path)
        tokenizer.push_to_hub(full_repo_name)
        del tokenizer
        free_memory()
        
        print("Loading model with 4-bit quantization on GPU...")
        
        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        # Load model with GPU-only settings
        model = AutoModelForCausalLM.from_pretrained(
            local_model_path,
            quantization_config=quantization_config,
            device_map="cuda",  # Force everything to GPU
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True
        )
        
        print("Pushing model (this may take a while)...")
        model.push_to_hub(
            full_repo_name,
            max_shard_size="100MB",
            use_auth_token=True
        )
        
        # Cleanup
        del model
        free_memory()
        
        print(f"Successfully pushed to: https://huggingface.co/{full_repo_name}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        print("\nDetailed error information:")
        import traceback
        print(traceback.format_exc())
        raise e
    finally:
        free_memory()

def verify_token():
    """Verify HuggingFace token"""
    try:
        user_info = whoami()
        print(f"Logged in as: {user_info['name']}")
        return user_info['name']
    except Exception as e:
        print("Token verification failed. Please ensure you're logged in.")
        raise e

if __name__ == "__main__":
    # Print system information
    print("\nSystem Information:")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1024**2:.0f}MB")
    print(f"Available GPU Memory: {torch.cuda.mem_get_info()[0]/1024**2:.0f}MB")
    
    push_to_hub(
        local_model_path="./final_model",
        repo_name="mistral-7b-risk-profile"  # Change this to your desired name
    )