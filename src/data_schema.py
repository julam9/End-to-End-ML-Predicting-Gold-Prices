from pydantic import BaseModel
from typing import Optional
from datetime import date

class GoldInput(BaseModel):
    """
    Data model for the gold input data.
    """
    start_date: Optional[date]
    end_date: Optional[date]