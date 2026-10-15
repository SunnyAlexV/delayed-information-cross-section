"""Run run_all.py's reproduction check in resumable batches.

WHY THIS EXISTS.  run_all.py --check is one ~75-minute process, and this
container reaps a background process whenever the session goes idle: two full
runs died silently, at lab07 and at lab21, with nothing in the log to say so.
A check that cannot survive its own environment is a check that does not run.

So the same work is done here in batches that fit inside one foreground call,
with the verdict for each lab appended to a state file as soon as it is known.
Nothing about the check itself changes: the lab list, the command, the
normalisation and the comparison are all imported from run_all, so this cannot
drift from what run_all does - it IS what run_all does, paused between labs.

    python batch_check.py --data <folder> [--budget 480]

Run it again to continue; it stops when the next lab would not fit in the
budget, and prints a summary when every lab has a verdict.
"""
import json
import os
import sys
import time

import run_all as R

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "batch_check_state.json")


def load():
    if os.path.exists(STATE):
        with open(STATE, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def save(st):
    with open(STATE, "w", encoding="utf-8") as fh:
        json.dump(st, fh, indent=1, sort_keys=True)


def main(argv):
    data = R._data_folder(argv)
    budget = 480.0
    for i, a in enumerate(argv):
        if a == "--budget" and i + 1 < len(argv):
            budget = float(argv[i + 1])
    st = load()
    os.makedirs(R.OUT, exist_ok=True)
    t0 = time.time()
    for script, arg, _what in R.ORDER:
        name = script.replace(".py", "")
        if name in st:
            continue
        if time.time() - t0 > budget:
            break
        text, code, secs = R.run(script, data if arg is not None else None)
        with open(os.path.join(R.OUT, name + ".txt"), "w", encoding="utf-8") as fh:
            fh.write(text)
        ref = os.path.join(R.EXPECTED, name + ".txt")
        if code != 0:
            tail = [x for x in text.strip().splitlines()[-4:]]
            st[name] = {"verdict": "EXITED NON-ZERO", "secs": round(secs),
                        "detail": tail}
        elif not os.path.exists(ref):
            st[name] = {"verdict": "NO REFERENCE", "secs": round(secs)}
        else:
            with open(ref, encoding="utf-8") as fh:
                got, want = R.normalise(text), R.normalise(fh.read())
            if got == want:
                st[name] = {"verdict": "MATCHES", "secs": round(secs),
                            "lines": len(got)}
            else:
                bad = [(i, w, g) for i, (w, g) in
                       enumerate(zip(want, got), 1) if w != g]
                st[name] = {"verdict": "DIFFERS", "secs": round(secs),
                            "differing_lines": len(bad),
                            "first": [{"line": i, "expected": w, "got": g}
                                      for i, w, g in bad[:3]],
                            "line_counts": [len(want), len(got)]}
        save(st)
        print(f"{st[name]['verdict']:>16}  {name}  ({st[name]['secs']}s)", flush=True)

    done = len(st)
    total = len(R.ORDER)
    print(f"\n{done} of {total} labs have a verdict")
    if done == total:
        by = {}
        for k, v in st.items():
            by.setdefault(v["verdict"], []).append(k)
        for verdict in sorted(by):
            print(f"  {verdict}: {len(by[verdict])}")
            if verdict != "MATCHES":
                for k in sorted(by[verdict]):
                    print(f"      {k}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
