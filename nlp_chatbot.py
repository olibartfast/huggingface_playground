from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")

# Initialize the pipeline
chatbot = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# Function to format the conversation
def format_conversation(messages):
    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

# Define the conversation
conversation = [
    {"role": "user", "content": "What are some fun activities I can do in the winter?"}
]


print("User:", conversation[0]['content'])

# Format the conversation
formatted_conversation = format_conversation(conversation)

# Generate a response
response = chatbot(formatted_conversation, max_length=100)

# Append the response to the conversation
conversation.append({"role": "assistant", "content": response[0]['generated_text']})

print("Assistant:", response[0]['generated_text'])

# Continue the conversation
conversation.append(
    {"role": "user", "content": "What else do you recommend?"}
)

print("User:", conversation[-1]['content'])

# Format the updated conversation
formatted_conversation = format_conversation(conversation)

# Generate a response
response = chatbot(formatted_conversation, max_length=100)

# Append the response to the conversation
conversation.append({"role": "assistant", "content": response[0]['generated_text']})

print("Assistant:", response[0]['generated_text'])
