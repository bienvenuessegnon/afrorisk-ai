"""Contrats d'entrée du endpoint de prédiction."""

from typing import Literal

from pydantic import BaseModel, Field

ContractType = Literal["Cash loans", "Revolving loans"]
EducationType = Literal[
    "Secondary / secondary special",
    "Higher education",
    "Incomplete higher",
    "Lower secondary",
    "Academic degree",
]
FamilyStatus = Literal[
    "Single / not married",
    "Married",
    "Civil marriage",
    "Widow",
    "Separated",
]
IncomeType = Literal[
    "Working",
    "Commercial associate",
    "Pensioner",
    "State servant",
    "Unemployed",
]
YesNo = Literal["Y", "N"]


class BorrowerInput(BaseModel):
    amt_income_total: float = Field(..., gt=0, le=100_000_000, examples=[120000])
    amt_credit: float = Field(..., gt=0, le=1_000_000_000, examples=[450000])
    amt_annuity: float = Field(..., gt=0, le=100_000_000, examples=[22000])
    amt_goods_price: float = Field(..., gt=0, le=1_000_000_000, examples=[400000])
    age_years: float = Field(..., ge=18, le=100, examples=[36])
    employment_years: float | None = Field(None, ge=0, le=70, examples=[6])
    ext_source_1: float | None = Field(None, ge=0, le=1, examples=[0.55])
    ext_source_2: float | None = Field(None, ge=0, le=1, examples=[0.62])
    ext_source_3: float | None = Field(None, ge=0, le=1, examples=[0.48])
    name_contract_type: ContractType
    name_education_type: EducationType
    name_family_status: FamilyStatus
    name_income_type: IncomeType
    cnt_children: int = Field(0, ge=0, le=20)
    flag_own_realty: YesNo
    flag_own_car: YesNo = "N"
    region_rating_client: int = Field(..., ge=1, le=3, examples=[2])
