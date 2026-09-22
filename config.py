"""Shared setup. Import this before `import cognee` in every script.

Loads .env, keeps Cognee's local databases inside this folder, builds the Bedrock model
for Strands. No API keys anywhere: both SDKs authenticate through your AWS profile.
"""

import os
import warnings
from pathlib import Path

import litellm
from dotenv import load_dotenv
from strands.models import BedrockModel

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

# Cognee stores its graph + vector DBs here instead of inside site-packages.
os.environ.setdefault("DATA_ROOT_DIRECTORY", str(ROOT / ".data_storage"))
os.environ.setdefault("SYSTEM_ROOT_DIRECTORY", str(ROOT / ".cognee_system"))
os.environ.setdefault("CACHE_ROOT_DIRECTORY", str(ROOT / ".cognee_cache"))

litellm.suppress_debug_info = True  # hide LiteLLM's red retry banner during recall
warnings.filterwarnings("ignore", category=RuntimeWarning)  # cognee-internal asyncio noise


def bedrock_model() -> BedrockModel:
    return BedrockModel(
        model_id=os.environ["STRANDS_MODEL_ID"],
        region_name=os.environ["AWS_REGION"],
    )
