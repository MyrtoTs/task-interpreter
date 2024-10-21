from config import *
from guidance import assistant, gen, models,system, user
import json
import os
import pandas as pd
import logging

def handle_messages(gpt, recipient, messages):
    with system():
        lm = gpt + recipient.system_message

    for message in messages:
        if message.get("role") == "user":
            with user():
                lm += message.get("content")
        else:
            with assistant():
                lm += message.get("content")
    return lm

def generate_response(lm):
    with assistant():
        lm += gen(name="initial_response")
    return lm["initial_response"]

def verify_json_response(lm):
    with user():
        lm += "Does the very last response from you contain JSON object? Respond with yes or no."
    with assistant():
        lm += gen(name="contains_json")

    if "yes" in lm["contains_json"].lower():
        with user():
            lm += (
                "What was that JSON object? Only respond with that valid JSON string. A valid JSON string starts with {"
            )
        with assistant():
            lm += "{" + gen(name="json")
        response = "{" + lm["json"]
        # verify that the response is valid json
        try:
            response_obj = structure.model_validate_json(response)
            return response_obj.model_dump_json(), True
        except Exception as e:
            return str(e), False
    return lm["initial_response"], False

def generate_structured_class(recipient, messages, sender, config):
    gpt = models.OpenAI("gpt-3.5-turbo-0125", api_key=llm_config_35.get("api_key"), echo=False)
    lm = handle_messages(gpt, recipient, messages)
    response = generate_response(lm)
    verified_response, is_valid_json = verify_json_response(lm)

    return True, verified_response if is_valid_json else response

def configure_logging(filename, level, format):
    logging.basicConfig(filename=filename, level=level, format=format)

def process_requests_and_log_to_excel(dataset_path, classifier_agent, output_file):
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logging.error(f"Dataset file not found: {dataset_path}")
        return

    # Read the dataset
    with open(dataset_path, 'r') as file:
        try:
            dataset = json.load(file)
        except json.JSONDecodeError as e:
            logging.error(f"Error parsing JSON file: {str(e)}")
            return

    # Prepare results list to store the processed results
    results = []

    # Process each request in the dataset
    for entry in dataset:
        user_input = entry.get('request', '')
        expected_category = entry.get('category', 'None')

        # Determine if request has an image context or not
        has_image = expected_category in ["IMAGE_RETRIEVAL_BY_IMAGE", "BINARY_VISUAL_QA", "IMAGE_SEGMENTATION", "OBJECT_COUNTING"]
        try:
            # Call the classifier agent to process the request
            request_existence, detected_category = classifier_agent.request_existence_and_classification(user_input, contains_image=has_image)

            # Append the result to the list
            results.append({
                'Request': user_input,
                'Expected Category': expected_category,
                'Detected Category': detected_category,
                'Request Exists': request_existence
            })

        except Exception as e:
            # Log the error in a log file and add an error entry to the results
            logging.error(f"Error processing request '{user_input}': {str(e)}")
            results.append({
                'Request': user_input,
                'Expected Category': expected_category,
                'Detected Category': 'Error',
                'Request Exists': 'Error'
            })

    # Convert results to a DataFrame
    df = pd.DataFrame(results)

    # Save the results to an Excel file
    df.to_excel(output_file, index=False)
    print(f"Results have been saved to {output_file}")

