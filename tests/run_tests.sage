r"""
Run the test suite. From the repository root::

    sage tests/run_tests.sage                   # the quick tests
    sage tests/run_tests.sage --long            # quick and long
    sage tests/run_tests.sage 'closed/*'        # test selection (shell glob)
    sage tests/run_tests.sage --list            # print names, nothing run
    sage tests/run_tests.sage --time out.json   # record the times

Each ``tests/test_*.sage`` file defines ``TESTS``, a dict ``name ->
(callable, tags)``, the tags taken from ``"quick"`` and ``"long"``. A test
raises ``AssertionError`` when it fails and returns nothing; it raises
``SkipTest`` when it cannot run here, which does not make the run fail.
"""
import argparse
import fnmatch
import json
import os
import platform
import statistics
import subprocess
import sys
from datetime import datetime
from time import perf_counter

load("tests/harness.sage")

ALL_TESTS = {}
for _file in sorted(f for f in os.listdir("tests")
                    if f.startswith("test_") and f.endswith(".sage")):
    TESTS = {}
    load("tests/" + _file)
    ALL_TESTS.update(TESTS)


def select(patterns, long):
    r"""
    The names of the quick tests, and of the long ones if ``long``, matching
    one of ``patterns`` if any.
    """
    names = [name for name, (_, tags) in ALL_TESTS.items()
             if "quick" in tags or (long and "long" in tags)]
    if patterns:
        names = [name for name in names
                 if any(fnmatch.fnmatch(name, p) for p in patterns)]
    return names


def run(names, repeat=1):
    r"""
    Run the named tests and return their results, printing each.
    """
    results = []
    width = max(len(name) for name in names) + 2
    for name in names:
        test = ALL_TESTS[name][0]
        times, error, skipped = [], None, None
        for _ in range(repeat):
            start = perf_counter()
            try:
                test()
            except SkipTest as exc:
                skipped = str(exc)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            times.append(perf_counter() - start)
            if error or skipped:
                break
        status = "skip" if skipped else ("FAIL" if error else "pass")
        seconds = float(statistics.median(times))
        results.append({"name": name, "status": status, "seconds": seconds,
                        "error": error})
        print(f"{status:>5s}  {name:{width}s} {seconds:7.2f}s", flush=True)
        if error or skipped:
            print(f"       {error or skipped}", flush=True)
    return results


def provenance():
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = None
    return {"commit": commit, "date": datetime.now().isoformat(timespec="seconds"),
            "machine": platform.node(), "cpus": os.cpu_count()}


def main(argv):
    parser = argparse.ArgumentParser(prog="run_tests.sage")
    parser.add_argument("patterns", nargs="*", help="shell globs on test names")
    parser.add_argument("--long", action="store_true", help="add the long tests")
    parser.add_argument("--list", action="store_true", help="list and exit")
    parser.add_argument("--time", metavar="OUT.json", help="record the times")
    parser.add_argument("--repeat", type=int, default=int(1),
                        help="runs per test, median kept")
    args = parser.parse_args(argv)
    names = select(args.patterns, args.long)
    if args.list:
        print("\n".join(names))
        return 0
    if not names:
        print("no test selected")
        return 1
    print(f"{len(names)} tests\n")
    results = run(names, args.repeat)
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print("\n" + "  ".join(f"{n} {s}" for s, n in sorted(counts.items()))
          + f"   in {sum(r['seconds'] for r in results):.1f}s")
    if args.time:
        with open(args.time, "w") as f:
            json.dump({"provenance": provenance(), "repeat": args.repeat,
                       "tests": {r["name"]: r["seconds"] for r in results},
                       "status": {r["name"]: r["status"] for r in results}},
                      f, indent=1, sort_keys=True)
            f.write("\n")
    return 1 if counts.get("FAIL") else 0


if os.path.basename(sys.argv[0]).startswith("run_tests"):
    _status = main(sys.argv[1:])
    # Sage turns sys.exit() into an exit status of 1 whatever the code
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(_status)
