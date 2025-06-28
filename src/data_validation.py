import pandera as pa
from pandera import Column, DataFrameSchema, Check

# Define schema for gold price data
gold_schema = DataFrameSchema({
    "Open": Column(pa.Float, Check.ge(0)),
    "High": Column(pa.Float, Check.ge(0)),
    "Low": Column(pa.Float, Check.ge(0)),
    "Close": Column(pa.Float, nullable=True),
    "Volume": Column(pa.Int, Check.ge(0)),
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