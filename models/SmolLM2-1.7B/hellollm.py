from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
device = "cpu"  # Change to "cuda" if using a GPU

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# # Get user input
prompt = input("Enter your prompt: ")

# prompt = "Who are you?"
messages = [{"role": "user", "content": prompt}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
# Prepare input
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# # Generate output with improved settings
outputs = model.generate(inputs, 
                         max_new_tokens=250, 
                         temperature=1.0, 
                         top_p=0.9, 
                         do_sample=True)

# # Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\n##################################")
print("Model Response:")
print("##################################\n")
print(response)
