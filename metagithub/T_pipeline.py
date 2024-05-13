import transformers
import torch
import os

model_id = "/Users/linyi/CodeMan/opensource/Llama3/Meta-Llama-3-8B-Instruct"

os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'

pipeline = transformers.pipeline(
    "text-generation",
    model=model_id,
    model_kwargs={"torch_dtype": torch.float32},
    device="mps",
)

messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

prompt = pipeline.tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
)

terminators = [
    pipeline.tokenizer.eos_token_id,
    pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

#调试
print("pipeline.tokenizer.eos_token_id 的值",pipeline.tokenizer.eos_token_id)
print("pipeline.tokenizer.convert_tokens_to_ids() 的值",pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"))
print("terminators 的值",terminators)
outputs = pipeline(
    prompt,
    max_new_tokens=256,
    eos_token_id=terminators,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
print(outputs[0]["generated_text"][len(prompt):])