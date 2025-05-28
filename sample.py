from lora_attention_analyzer import LoRAAttentionPipeline

pipeline = LoRAAttentionPipeline("your_model_file.safetensors", model_type="epsilon") # model_type: epsilon, vpred
results = pipeline.run_comparison(
    lora_file="your_lora_file.safetensors",
    prompt="your_prompt",
    tokens_to_analyze=["token1", "token2", "token3"],
    output_dir="output"
)