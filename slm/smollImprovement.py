from transformers import AutoModelForCausalLM, AutoTokenizer

import re
import numpy as np
import matplotlib.pyplot as plt
from fABBA import fABBA

sequenceLenght = 1000

######### STEP 1: GET ABBA DATA ############
ts = [np.sin(0.05*i) for i in range(sequenceLenght)]  # original time series
fabba = fABBA(tol=0.1, alpha=0.1, sorting='2-norm', scl=1, verbose=0)

dataString = fabba.fit_transform(ts)            # string representation of the time series
sequenceLenghtPerToken = sequenceLenght/len(dataString)

print(f"fABBA Original {dataString}")

######## STEP 2: PREPARE THE SLM ############

checkpoint = "HuggingFaceTB/SmolLM2-135M-Instruct"
device = "cpu"  # Change to "cuda" if using a GPU

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)

##### STEP 3: PREPARE DATA REQUEST ##########

# # Get user input
dataPredictionrequest = f"I will present you a time series data that is masked as lower and upper case letters. I want you to predict the sequence of following letters."

messages = [{"role": "user", "content": dataPredictionrequest}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
# Prepare input
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# # Generate output with improved settings
outputs = model.generate(inputs, 
                         max_new_tokens=100, 
                         temperature=1.0, 
                         top_p=0.9, 
                         do_sample=False)

# # Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\n##################################")
print("Model Response:")
print("##################################\n")
print(response)

######## STEP 4: INSERT DATA #################

messages = [{"role": "user", "content": dataString}]
input_text=tokenizer.apply_chat_template(messages, tokenize=False)
# Prepare input
inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)

# # Generate output with improved settings
outputs = model.generate(inputs, 
                         max_new_tokens=40, 
                         temperature=1.0, 
                         top_p=0.9, 
                         do_sample=False)

# # Decode and print response
response = tokenizer.decode(outputs[0], skip_special_tokens=False)
print("\n##################################")
print("Model Response:")
print("##################################\n")
print(response)

matches = re.findall(r"<\|im_start\|>assistant\n(.*?)(?=(<\|im_start\|>|$))", response, re.DOTALL)

if matches:
    last_response = matches[-1][0].strip()
    print(last_response)
else:
    print("No assistant response found.")
    exit

responeCleaned = re.sub(r'[^a-zA-Z]', '', last_response)


print("#################333")
print(responeCleaned)
print(dataString+responeCleaned)


# ######## STEP 5: MODEL BACK #################

inverse_ts = fabba.inverse_transform(dataString, ts[0]) # numerical time series reconstruction
inverse_ts_predicted = fabba.inverse_transform(responeCleaned, ts[999]) # predicted

# print(inverse_ts_predicted)

plt.plot(ts, label='time series')
plt.plot(inverse_ts, label='reconstruction')
plt.plot(range(1000, 1000 + len(inverse_ts_predicted[1000:])), inverse_ts_predicted[1000:], label='Prediction')
plt.legend()
plt.grid(True, axis='y')
plt.show()
plt.savefig("reconstructionPrediction.png")
