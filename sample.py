from lora_attention_analyzer import LoRAAttentionPipeline

pipeline = LoRAAttentionPipeline("/data3/ComfyUI/models/checkpoints/Illustrious-XL-v3.5-vpred.safetensors", model_type="vpred") # model_type: epsilon, vpred
results = pipeline.run_comparison(
    lora_adaptive_file="/data6/dh/lora_attn_map/loras/pochacco_lora-time.safetensors",
    lora_normal_file="/data6/dh/lora_attn_map/loras/pochacco_lora-only.safetensors",
    prompt= "pochacco, aesthetic, flower field, no humans, colorful flowers, solo, sitting, outdoors, day, butterflies, rainbow border", 
    tokens_to_analyze=["pochacco", "aesthetic", "flower field", "no humans", "colorful flowers", "solo", "sitting", "outdoors", "day", "butterflies", "rainbow border"],
    output_dir="./output_0624_blog/pochacco_1_v8",
    negative_prompt="worst quality, low quality, lowers, low details, bad quality, poorly drawn, bad anatomy, multiple views, bad hands, blurry, artist sign, weibo username", # if negative_prompt is None: default negative prompt is used
    lora_scale=1.0,
    seed=0, 
    steps=28,
    guidance_scale=7.5, # cfg scale
    height=1024,
    width=1024,
)