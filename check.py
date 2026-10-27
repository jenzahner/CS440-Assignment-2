"""Check hw2.ipynb from a terminal (the same as check() inside the notebook).

    python3 check.py                 # everything
    python3 check.py --part p1       # one problem: p1, p2 or p3

It reads the SAVED notebook, runs the functions in its GRADED cells on example
inputs, and checks the format of the answers.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from hw2lib.grading import PARTS  # noqa: E402
from hw2lib.notebook import check  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=list(PARTS))
    ap.add_argument("--notebook", default=os.path.join(HERE, "hw2.ipynb"))
    args = ap.parse_args()
    check(args.part, args.notebook)
