"""
make_all.py - one command that rebuilds everything and checks everything.

    python make_all.py --data <folder>          full run: labs, papers, checks
    python make_all.py --data <folder> --fast   skip the lab run, check the rest

WHY THIS FILE EXISTS
--------------------
Three commands had to be run in the right order, and the order was only written
down in the README.  Running verify_paper.py before run_all.py compares the
paper against yesterday's output and passes when it should fail, which is the
one failure mode a verifier must not have.  This file removes the ordering from
the reader's hands:

    1. verify_data.py    is your copy of the inputs ours?
    2. run_all.py        do the scripts still produce the stored output?
    3. build_pdf.py      render the three documents from their source
    4. verify_paper.py   does every figure in them match that output?
    5. build_variants.py regenerate the journal submissions from that source
    6. build_ssrn.py     bind the SSRN copy and check both seams
    7. build_registry.py the five gates over every quoted number
    8. verify_letters.py do the cover letters still match the package?
    9. verify_repo.py    does only the reproduction ship, described honestly?

Step 2 takes about two hours.  --fast skips it, which is the right choice while
editing prose and the wrong choice before shipping, so the summary says plainly
which of the two was run.  Any step failing stops the rest, because a later
step's verdict would be meaningless.
"""

import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def step(name, argv, why):
    print(f"\n{'=' * 78}\n{name}\n  {why}\n{'=' * 78}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable] + argv, cwd=HERE)
    secs = time.time() - t0
    ok = r.returncode == 0
    print(f"\n  {name}: {'OK' if ok else 'FAILED'} ({secs:.0f}s)", flush=True)
    return ok, secs


def main(argv):
    fast = "--fast" in argv
    data = None
    for i, a in enumerate(argv):
        if a == "--data" and i + 1 < len(argv):
            data = argv[i + 1]
        elif a.startswith("--data="):
            data = a.split("=", 1)[1]

    plan = []
    if data:
        plan.append(("verify_data", ["verify_data.py", data],
                     "is your copy of the inputs the one the numbers came from?"))
    else:
        print("no --data given, so the input check and the lab run are skipped;\n"
              "the papers are still rendered and checked against the STORED output.")
    if not fast and data:
        plan.append(("run_all --check", ["run_all.py", "--check", "--data", data],
                     "do all 59 scripts still produce the stored output?"))
    plan.append(("build_pdf", ["build_pdf.py"],
                 "render the three documents from their HTML source"))
    plan.append(("verify_paper", ["verify_paper.py"],
                 "does every figure in them match that output?"))
    # The journal variants are generated from the same source, so they go stale
    # the moment the source is corrected and nobody rebuilds them.  Putting the
    # build here means a correction cannot reach papers/ without reaching the
    # submissions as well; the step re-verifies each variant itself, so it
    # fails rather than quietly shipping three documents that disagree.
    plan.append(("build_variants", ["build_variants.py"],
                 "regenerate the three journal variants and verify each"))
    plan.append(("build_ssrn", ["build_ssrn.py"],
                 "bind the paper and its appendix into the one PDF SSRN takes"))
    plan.append(("build_registry", ["build_registry.py"],
                 "the five build gates over every number that is quoted"))
    # The cover letters quote the package's own counts, and nobody re-derives a
    # cover letter.  Checking them here is what stops a letter telling an
    # editor the paper has 1,409 checks after it has 1,772.
    plan.append(("verify_letters", ["verify_letters.py"],
                 "do the cover letters still describe the package they ship with?"))
    plan.append(("verify_repo", ["verify_repo.py"],
                 "does only the reproduction ship, and does the README describe it?"))

    results = []
    for name, cmd, why in plan:
        ok, secs = step(name, cmd, why)
        results.append((name, ok, secs))
        if not ok:
            print(f"\n{'=' * 78}\nSTOPPED at {name}. The steps after it would report on "
                  f"a state that\nhas not been rebuilt, so they are not run.\n{'=' * 78}")
            return 1

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    for name, ok, secs in results:
        print(f"  {'OK    ' if ok else 'FAILED'}  {name:<18} {secs:>6.0f}s")
    if fast or not data:
        print("\n  The lab run was SKIPPED, so this says the papers agree with the\n"
              "  STORED output, not that the scripts still produce it. Before\n"
              "  shipping, run without --fast.")
    else:
        print("\n  Inputs verified, all 59 scripts reproduced, both documents "
              "rendered,\n  and every checked figure in them matches. Nothing is "
              "outstanding.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
