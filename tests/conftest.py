import shutil
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

needs_rubberband = pytest.mark.skipif(
    shutil.which("rubberband") is None or shutil.which("ffmpeg") is None,
    reason="needs the rubberband and ffmpeg command-line tools",
)
