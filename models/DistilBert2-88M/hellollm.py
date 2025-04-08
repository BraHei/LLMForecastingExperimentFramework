from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "distilbert/distilgpt2"
device = "cuda"  # Change to "cuda" if using a GPU

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

# # Get user input
prompt = input("Enter your prompt: ")

# Prepare input
inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)

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
