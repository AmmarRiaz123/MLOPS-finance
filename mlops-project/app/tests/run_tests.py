"""
Test runner for app routes.

Usage examples (from mlops-project folder):
  - Run all tests:
      python -m app.tests.run_tests --all

  - Run a specific test file:
      python -m app.tests.run_tests --file test_direction.py

  - Run a specific test (pytest command shown and executed):
      python -m app.tests.run_tests --pytest "app/tests/test_direction.py::test_predict_direction_route_exists"

This script will also print equivalent pytest commands you can run manually:
  pytest app/tests/test_root_health.py -q
  pytest app/tests/test_direction.py -q
  pytest app/tests/test_all_routes.py -q
"""
import argparse
import subprocess
import sys

DEFAULT_TEST_FILES = [
    "app/tests/test_root_health.py",
    "app/tests/test_direction.py",
    "app/tests/test_all_routes.py",
]

def run_pytest(args_list):
    cmd = [sys.executable, "-m", "pytest"] + args_list
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--all", action="store_true", help="Run all tests")
    p.add_argument("--file", help="Run a specific test file (filename in app/tests)")
    p.add_argument("--pytest", help="Run a pytest target (full pytest argument)")
    args = p.parse_args()

    print("\nExample pytest commands (run manually):")
    for f in DEFAULT_TEST_FILES:
        print(f"  pytest {f} -q")
    print("  pytest app/tests/test_direction.py::test_predict_direction_route_exists -q\n")

    if args.all:
        sys.exit(run_pytest(DEFAULT_TEST_FILES))
    if args.file:
        sys.exit(run_pytest([f"app/tests/{args.file}"]))
    if args.pytest:
        sys.exit(run_pytest([args.pytest]))

    p.print_help()

if __name__ == "__main__":
    main()
