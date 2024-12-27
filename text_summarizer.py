import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Step 1: Initialize the model and tokenizer
def load_model_and_tokenizer():
    """Load the T5 model and tokenizer, and set the device (CPU for this example). """

    print("Loading the T5 model and tokenizer...")
    model = T5ForConditionalGeneration.from_pretrained('t5-small')  # Load the T5 model
    tokenizer = T5Tokenizer.from_pretrained('t5-small')  # Load the T5 tokenizer
    device = torch.device('cpu')  # Use CPU for this script
    print("Model and tokenizer loaded successfully!")
    return model, tokenizer, device

# Step 2: Preprocess the input text
def prepare_input_for_summarization(text):
    """ Clean and prepare the input text for the T5 summarization task. """

    # Remove unnecessary newlines and add the task-specific prefix
    cleaned_text = text.strip().replace('\n', ' ')
    task_prefixed_text = "summarize: " + cleaned_text
    return task_prefixed_text

# Step 3: Perform text summarization
def generate_summary(input_text, model, tokenizer, device, min_len=30, max_len=120):
    """ Generate a summary for the given text using the T5 model. """

    print("Preprocessing the input text...")
    prepared_text = prepare_input_for_summarization(input_text)
    
    print("Encoding the text for the model...")
    # Convert the text into token IDs
    input_ids = tokenizer.encode(prepared_text, return_tensors='pt', max_length=512, truncation=True).to(device)
    
    print("Generating the summary...")
    # Generate the summary using the T5 model
    summary_ids = model.generate(input_ids, min_length=min_len, max_length=max_len, length_penalty=2.0, num_beams=4)
    
    print("Decoding the summary back to text...")
    # Decode the generated token IDs into readable text
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Step 4: Main function to interact with the user
if __name__ == "__main__":
    print("Welcome to the Text Summarizer!")
    print("This program will help you generate a concise summary of any text you provide.")

    # Load the model, tokenizer, and device
    model, tokenizer, device = load_model_and_tokenizer()

    # Ask the user for input text and summary length preferences
    input_text = input("\nPlease enter the text you want to summarize:\n")
    min_length = input("\nEnter the minimum summary length (default is 30): ")
    max_length = input("Enter the maximum summary length (default is 120): ")

    # Handle optional user inputs for lengths
    try:
        min_length = int(min_length) if min_length.strip() else 30
        max_length = int(max_length) if max_length.strip() else 120
    except ValueError:
        print("Invalid length values entered. Using default values: 30 (min), 120 (max).")
        min_length = 30
        max_length = 120

    # Generate the summary
    print("\nSummarizing the text...")
    summary_result = generate_summary(input_text, model, tokenizer, device, min_length, max_length)

    # Display the summary
    print("\nHere is your summary:\n")
    print(summary_result)

    print("\nThank you for using the Text Summarizer!")
