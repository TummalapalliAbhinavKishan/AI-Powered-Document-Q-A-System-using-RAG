import sys
import os

_here = os.path.dirname(os.path.abspath(__file__))
for _p in [
    _here,
    os.path.normpath(os.path.join(_here, "..")),
    os.path.normpath(os.path.join(_here, "..", "..")),
    os.path.normpath(os.path.join(_here, "..", "..", "..")),
    "/var/task",
]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from mangum import Mangum
from main import app  # noqa: E402

handler = Mangum(app, lifespan="off")
