from pydantic import BaseModel, Field
from typing import Optional

class CreditInputSchema(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., ge=0.0)
    age: int = Field(..., ge=0, le=120)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime30-59DaysPastDueNotWorse", ge=0)
    DebtRatio: float = Field(..., ge=0.0)
    MonthlyIncome: Optional[float] = Field(None, ge=0.0)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., ge=0)
    NumberOfTimes90DaysLate: int = Field(..., ge=0)
    NumberRealEstateLoansOrLines: int = Field(..., ge=0)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., alias="NumberOfTime60-89DaysPastDueNotWorse", ge=0)
    NumberOfDependents: Optional[float] = Field(None, ge=0.0)

    class Config:
        populate_by_name = True


class CreditPredictionResponse(BaseModel):
    probability: float
    prediction: int
    risk_label: str
    confidence_level: str
