def request_existence_and_classification(user_input, has_image=True):
    categories = {
        True: ['IMAGE_RETRIEVAL_BY_IMAGE', 'VISUAL_QA', 'IMAGE_SEGMENTATION', 'OBJECT_COUNTING', 'None'],
        False: ['IMAGE_RETRIEVAL_BY_CAPTION', 'IMAGE_RETRIEVAL_BY_METADATA', 'GEOGRAPHY_QA', 'None']
    }
    
    category_list = ', '.join(categories[has_image])
    input_type = "satellite image" if has_image else "input"
    message = f"""
    Possible request types are: {category_list}.
    Label the USER_INPUT via the following instructions:

    {config.response_prompt_instructions}

    USER_INPUT: Given this {input_type}:{user_input}
    """
    
    user_proxy.initiate_chat(
        guidance_agent,
        max_turns=1,
        message=message
    )
    
    response_json = json.loads(user_proxy.chat_messages[guidance_agent][-1]["content"])
    return response_json["request_existence"], response_json["request_category"]