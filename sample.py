from lora_attention_analyzer import LoRAAttentionPipeline

pipeline = LoRAAttentionPipeline("your_model_file.safetensors", model_type="epsilon") # model_type: epsilon, vpred
results = pipeline.run_comparison(
    lora_file="your_lora_file.safetensors",
    prompt="your_prompt",
    tokens_to_analyze=["token1", "token2", "token3"],
    output_dir="output",
    negative_prompt="your_negative_prompt", # if negative_prompt is None: default negative prompt is used
    lora_scale=1.0,
    seed=0,
    steps=50,
    guidance_scale=7.5,
    height=1024,
    width=1024,
)