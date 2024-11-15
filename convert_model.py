from transformers import LlamaTokenizer, LlamaForCausalLM, PretrainedConfig, LlamaConfig
import torch
import json
import os
import shutil

def convert_model(input_path, output_path):
    print(f"Converting model from {input_path} to {output_path}")
    
    # First, verify the input path contains the expected files
    required_files = ['tokenizer.model', 'consolidated.00.pth', 'params.json']
    for file in required_files:
        if not os.path.exists(os.path.join(input_path, file)):
            raise FileNotFoundError(f"Missing required file: {file} in {input_path}")
    
    # Load the params.json to get model configuration
    with open(os.path.join(input_path, 'params.json'), 'r') as f:
        params = json.load(f)
        print(f"Model parameters: {params}")
    
    # Load the consolidated.00.pth file
    print("Loading model weights...")
    state_dict = torch.load(
        os.path.join(input_path, 'consolidated.00.pth'),
        map_location='cpu',
        weights_only=True
    )
    
    # Create a new state dict with remapped keys
    new_state_dict = {}
    
    # Map the keys from Meta format to HF format
    key_mapping = {
        "tok_embeddings.weight": "model.embed_tokens.weight",
        "norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }
    
    layer_mapping = {
        "attention.wq.weight": "self_attn.q_proj.weight",
        "attention.wk.weight": "self_attn.k_proj.weight",
        "attention.wv.weight": "self_attn.v_proj.weight",
        "attention.wo.weight": "self_attn.o_proj.weight",
        "feed_forward.w1.weight": "mlp.gate_proj.weight",
        "feed_forward.w2.weight": "mlp.down_proj.weight",
        "feed_forward.w3.weight": "mlp.up_proj.weight",
        "attention_norm.weight": "input_layernorm.weight",
        "ffn_norm.weight": "post_attention_layernorm.weight",
    }
    
    print("Remapping state dict keys...")
    for key in state_dict:
        if key in key_mapping:
            new_key = key_mapping[key]
            new_state_dict[new_key] = state_dict[key]
        elif key.startswith("layers."):
            # Handle layer-specific keys
            parts = key.split(".")
            layer_id = parts[1]
            sub_key = ".".join(parts[2:])
            
            if sub_key in layer_mapping:
                new_key = f"model.layers.{layer_id}.{layer_mapping[sub_key]}"
                new_state_dict[new_key] = state_dict[key]
    
    # Create the model config using LlamaConfig
    config = LlamaConfig(
        hidden_size=params["dim"],
        intermediate_size=int(params["dim"] * params["ffn_dim_multiplier"]),
        num_attention_heads=params["n_heads"],
        num_hidden_layers=params["n_layers"],
        num_key_value_heads=params["n_kv_heads"],
        rms_norm_eps=params["norm_eps"],
        rope_theta=params["rope_theta"],
        vocab_size=params["vocab_size"],
    )
    
    # Save the config
    config.save_pretrained(output_path)
    
    # Create model with this config
    model = LlamaForCausalLM(config)
    
    # Load the remapped state dict
    print("Loading remapped weights into model...")
    model.load_state_dict(new_state_dict, strict=True)
    
    # Save in HF format
    print("Saving model in Hugging Face format...")
    model.save_pretrained(output_path, max_shard_size="10GB")
    
    # Handle tokenizer
    print("Converting tokenizer...")
    shutil.copy2(
        os.path.join(input_path, "tokenizer.model"),
        os.path.join(output_path, "tokenizer.model")
    )
    
    tokenizer_config = {
        "model_max_length": 4096,
        "padding_side": "right",
        "use_fast": True,
        "tokenizer_class": "LlamaTokenizer",
    }
    
    with open(os.path.join(output_path, "tokenizer_config.json"), "w") as f:
        json.dump(tokenizer_config, f)
    
    print(f"Conversion complete! Model saved to {output_path}")

if __name__ == "__main__":
    input_path = r"C:\Users\ngyat\.llama\checkpoints\Llama3.1-8B"
    output_path = r"C:\Users\ngyat\.llama\checkpoints\Llama3.1-8B-hf"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    convert_model(input_path, output_path)