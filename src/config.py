import os
import logging

llm_config_35 = {
    "config_list": [{
        "model": "gpt-3.5-turbo-0125", 
        "api_key": os.getenv("OPENAI_API_KEY"),
        "temperature": 0.0,  # Set temperature to 0 for deterministic output
        "top_p": 1.0         # Set top_p to 1 for full distribution (can be left unchanged)
    }]
}

response_prompt_instructions = """The label must be a JSON of the format:
{
    "request_existence": bool,
    "explanation": str,
    "request_category": str
}"""

log_config = {
    "filename": "../results/classification_errors.log",
    "level": logging.ERROR,
    "format": '%(asctime)s - %(levelname)s - %(message)s'
}
