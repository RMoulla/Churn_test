from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class FeatureConfig:
    numerical: List[str]
    categorical: List[str]


FEATURES = FeatureConfig(
    numerical=["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"],
    categorical=[
        "gender",
        "Partner",
        "Dependents",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
    ],
)

TARGET_COLUMN = "Churn"
IDENTIFIER_COLUMN = "customerID"
