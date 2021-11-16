"""Data getter for Glass to Companies House matching."""
import io
import logging

import boto3
import pandas as pd


def get_glass_house() -> pd.DataFrame:
    """Gets matches between Glass and Companies house (and accompanying score)."""
    logging.warn("This Glass-House data is a temporary placeholder")

    s3 = boto3.client("s3")
    obj = s3.get_object(
        Bucket="nesta-glass", Key="data/processed/glass/company_numbers.csv"
    )
    return pd.read_csv(io.BytesIO(obj["Body"].read()))
