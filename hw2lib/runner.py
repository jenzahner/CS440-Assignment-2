"""Runs one test group in its own process. Used by check() and for grading.

    python3 hw2lib/runner.py GROUP CASES.json [--key KEY.json] [--testset TEST.npz]

The work directory (the parent of hw2lib/) must contain the student modules.
Prints one line starting with RESULT_MARK, holding the JSON result.
"""
import json
import os
import sys

RESULT_MARK = "@@HW2-RESULT@@"


def main(argv):
    group, case_file = argv[0], argv[1]
    key_file = argv[argv.index("--key") + 1] if "--key" in argv else None
    testset = argv[argv.index("--testset") + 1] if "--testset" in argv else None

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path[:0] = [root]
    os.chdir(root)
    from hw2lib import grading
    try:  # one thread: the same speed everywhere, and no slowdown from oversubscription
        import torch
        torch.set_num_threads(1)
    except ImportError:
        pass

    points_max = grading.GROUPS[group][0]
    try:
        if group.startswith("answers"):
            key = None
            if key_file:
                with open(key_file) as f:
                    key = json.load(f)
            points, messages = grading.grade_answers(group, key)
            result = {"points": points, "messages": messages}
        elif group == "open":
            points, messages = grading.group_open(testset)
            result = {"points": points, "messages": messages}
        else:
            with open(case_file) as f:
                data = json.load(f)
            passed, total, messages = grading.GROUP_FUNCTIONS[group](data)
            result = {"points": round(points_max * passed / total, 2), "passed": passed,
                      "total": total, "messages": messages}
    except Exception as exc:  # noqa: BLE001 - e.g. a syntax error in a student module
        result = {"points": 0, "messages": [f"could not run: {grading.describe(exc)}"]}
    result["max"] = points_max
    sys.stdout.write("\n" + RESULT_MARK + json.dumps(result) + "\n")
    sys.stdout.flush()


if __name__ == "__main__":
    main(sys.argv[1:])
