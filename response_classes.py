from pydantic import BaseModel
from typing import Literal

class Response_Image(BaseModel):
    request_existence: bool
    explanation: str
    request_category: Literal['IMAGE_RETRIEVAL_BY_IMAGE', 'VISUAL_QA', 'IMAGE_SEGMENTATION', 
                             'OBJECT_COUNTING', 'None']

class Response_NoImage(BaseModel):
    request_existence: bool
    explanation: str
    request_category: Literal['IMAGE_RETRIEVAL_BY_CAPTION', 'IMAGE_RETRIEVAL_BY_METADATA', 'GEOSPATIAL_QA', 'None']