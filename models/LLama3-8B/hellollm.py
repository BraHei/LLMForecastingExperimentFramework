import os
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

checkpoint = "meta-llama/Meta-Llama-3-8B-Instruct"
device = "cuda"  # Change to "cuda" if using a GPU

# Check if model is already downloaded
local_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub", "models--" + checkpoint.replace("/", "--"))

access_token = input("Enter your Hugging Face access token: ")
snapshot_download(repo_id=checkpoint, token=access_token)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint, token=access_token, torch_dtype="auto").to("cuda")

# Get user input
prompt = input("Enter your prompt: ")

messages = [{"role": "user", "content": prompt}]
input_text = tokenizer.apply_chat_template(messages, tokenize=False)

# Prepare input
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# Generate output
outputs = model.generate(
    inputs,
    max_new_tokens=250,
    temperature=1.0,
    top_p=0.9,
    do_sample=True
)

# Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\n##################################")
print("Model Response:")
print("##################################\n")
print(response)
