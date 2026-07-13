from pydantic import BaseModel, Field

class UserRequest(BaseModel):
    user_request: str = Field()

class ModelResponse(BaseModel):
    model_answer: str
