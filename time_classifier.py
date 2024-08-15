import spacy

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_time_range(question):
    # Process the input question using spaCy
    doc = nlp(question)

    # Initialize variables to store time entities
    start_time = None
    end_time = None

    # Loop through recognized entities in the question
    for ent in doc.ents:
        if ent.label_ in ["TIME", "DATE"]:
            # Assign start_time and end_time if recognized
            if start_time is None:
                start_time = ent.text
            else:
                end_time = ent.text

    # Check if both start_time and end_time were found
    if start_time and end_time:
        return f"Time Range: {start_time} to {end_time}"
    elif start_time:
        return f"Single Time Entity: {start_time}"
    else:
        return "No time range found."

# Example usage
question = "Which stock will grow the most in 7 weeks?"
time_range = extract_time_range(question)
print(time_range)
