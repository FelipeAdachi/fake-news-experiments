import os

ROOT_DIR = os.path.dirname(os.path.abspath(
    __file__))  # This is your Project Root

lakefs_conf = {
    "lakefs_key_id": "<your-lakefs-key-id>",
    "lakefs_secret_key": "<your-lakefs-secret-key>",
    "lakefs_url": "localhost:8000",
    "repository": "<your-repository>"
}

TEMP_DIR = "temp"
