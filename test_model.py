from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from peft import PeftModel, PeftConfig

def load_model():
    # Load the base model and apply the trained LoRA weights
    base_model_name = "mistralai/Mistral-7B-v0.1"
    trained_model_path = "training/dad-ai-model-final"
    
    # Configure 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        load_in_8bit=True,
        offload_folder="offload",
        offload_state_dict=True,
        low_cpu_mem_usage=True
    )
    
    # Load trained model with optimized settings
    model = PeftModel.from_pretrained(
        model, 
        trained_model_path,
        offload_folder="offload",
        device_map="auto"
    )
    
    # Enable model evaluation mode
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer

def generate_response(prompt, model, tokenizer):
    # More personal context emphasizing one-on-one father-son relationship
    context = "You are my loving and wise father, and I am your only son. Respond to me with fatherly love, understanding, and wisdom, showing that special bond we share. Remember our close relationship and speak to me as only a father can speak to his beloved son: "
    full_prompt = context + prompt
    
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        num_return_sequences=1,
        temperature=0.6,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        use_cache=True,
        eos_token_id=tokenizer.eos_token_id,
        num_beams=1
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response = response[len(full_prompt):].strip()
    
    # Ensure complete sentences
    if not response.rstrip().endswith(('.', '!', '?')):
        last_period = response.rfind('.')
        last_exclaim = response.rfind('!')
        last_question = response.rfind('?')
        
        last_end = max(last_period, last_exclaim, last_question)
        
        if last_end != -1:
            response = response[:last_end + 1]
        else:
            response = response.rstrip() + '.'
    
    return response

def main():
    print("Loading model...")
    model, tokenizer = load_model()
    
    print("\nModel loaded! You can now chat with your AI Dad.")
    print("Type 'quit' to exit the chat.")
    print("\nTip: Feel free to ask for advice, share your thoughts, or just chat!")
    
    while True:
        user_input = input("\nYou: ")
        if user_input.lower() == 'quit':
            break
        
        print("\nAI Dad is thinking...")  # Added to show it's processing
        response = generate_response(user_input, model, tokenizer)
        print(f"\nAI Dad: {response}")

if __name__ == "__main__":
    main() 