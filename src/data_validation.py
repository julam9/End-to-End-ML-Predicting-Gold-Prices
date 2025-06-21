import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Define schema for gold price data
gold_schema = DataFrameSchema({
    "open": Column(pa.Float, Check.ge(0)),
    "high": Column(pa.Float, Check.ge(0)),
    "low": Column(pa.Float, Check.ge(0)),
    "close": Column(pa.Float, nullable=True),
    "volume": Column(pa.Int, Check.ge(0)),
})

def validate_data(df):
    """
    Validate the input DataFrame using a predefined schema.

    Args:
        df (pd.DataFrame): The gold price DataFrame.

    Returns:
        pd.DataFrame: Validated DataFrame.
    """
    return gold_schema.validate(df)