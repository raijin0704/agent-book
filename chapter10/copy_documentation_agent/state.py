import operator
from typing import Annotated

from pydantic import BaseModel, Field

from .data_model import Interview, Persona


class InterviewState(BaseModel):
    user_request: str = Field(..., description="ユーザーからのリクエスト")
    personas: Annotated[list[Persona], operator.add] = Field(
        default_factory=list, description="生成されたペルソナのリスト"
    )
    interviews: Annotated[list[Interview], operator.add] = Field(
        default_factory=list, description="実施されたインタビューのリスト"
    )
    requirements_doc: str = Field(default="", description="生成された要件定義")
    iteration: int = Field(
        default=0, description="ペルソナ作成とインタビューの反復回数"
    )
    is_information_sufficient: bool = Field(
        default=False, description="情報が十分かどうか"
    )
