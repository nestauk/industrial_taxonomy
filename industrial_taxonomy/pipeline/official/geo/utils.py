import requests


def fetch_boundary(boundary_url: str) -> dict:
    """Fetch shapefile data"""

    return requests.get(boundary_url).json()
