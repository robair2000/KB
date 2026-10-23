"""Pre-fetch Kronos weights at build time so worker restarts don't hit Hugging Face."""
from huggingface_hub import snapshot_download

import config as C

for repo in (C.MODEL_NAME, C.TOKENIZER_NAME):
    print("downloading", repo, "->", snapshot_download(repo))
