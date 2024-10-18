import os
import logging

llm_config_35 = {
    "config_list": [{"model": "gpt-3.5-turbo-0125", "api_key": os.getenv("OPENAI_API_KEY")}],
}

response_prompt_instructions = """The label must be a JSON of the format:
{
    "request_existence": bool,
    "explanation": str,
    "request_category": str
}"""

log_config = {
    "filename": "results/classification_errors.log",
    "level": logging.ERROR,
    "format": '%(asctime)s - %(levelname)s - %(message)s'
}
