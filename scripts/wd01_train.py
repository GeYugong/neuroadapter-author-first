"""Reuse verified training and telemetry with a scoped weight-decay restore."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import balanced_train as entry
import wd01_resume
entry.balanced_resume=wd01_resume
if __name__=='__main__':entry.main()
