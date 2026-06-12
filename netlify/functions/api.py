import sys
import os

# Resolve project root so imports of main.py, data_loader.py, etc. work.
# Netlify bundles included_files at the same relative path inside the Lambda
# package, so the repo root ends up two levels above this file.
_here = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _here,                                        # function dir
    os.path.normpath(os.path.join(_here, "..", "..")),  # repo root
    "/var/task",                                  # Lambda working dir fallback
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mangum import Mangum
from main import app  # noqa: E402

handler = Mangum(app, lifespan="off")
