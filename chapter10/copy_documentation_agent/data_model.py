from pydantic import BaseModel, Field


class Persona(BaseModel):
    name: str = Field(..., description="ペルソナの名前")
    background: str = Field(..., description="ペルソナの持つ背景")


class Personas(BaseModel):
    personas: list[Persona] = Field(
        default_factory=list, description="インタビュー対象のペルソナ"
    )


class Interview(BaseModel):
    persona: Persona = Field(..., description="インタビュー対象のペルソナ")
    question: str = Field(..., description="インタビューでの質問")
    answer: str = Field(..., description="インタビューでの回答")


class Interviews(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビューの結果リスト"
    )


class InterviewResult(BaseModel):
    interviews: list[Interview] = Field(
        default_factory=list, description="インタビュー結果のリスト"
    )


class EvaluationResult(BaseModel):
    reason: str = Field(..., description="判断の理由")
    is_sufficient: bool = Field(..., description="情報が十分かどうか")
