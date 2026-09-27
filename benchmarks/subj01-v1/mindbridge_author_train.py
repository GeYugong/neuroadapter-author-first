"""Run the unchanged author entry point with workspace-local asset resolution."""
from pathlib import Path
import os
import runpy
import sys

from mindbridge_local_assets import configure

B = Path(__file__).resolve().parent
source = B / 'upstream/mindbridge/src'
configure()
os.chdir(source)
sys.path.insert(0, str(source))
runpy.run_path(str(source / 'main.py'), run_name='__main__')
