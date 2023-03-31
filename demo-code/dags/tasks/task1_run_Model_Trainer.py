"""FIRST TASK"""

import requests


def main():
    """Call the service via HTTP REST API."""
    result = requests.get("http://34.69.150.86/run")
    print(result.json())
