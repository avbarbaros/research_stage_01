"""CPU unit tests with tiny random models."""

import sys

from _common import run

run(sys.executable, "-m", "pytest", "tests", "-q", *sys.argv[1:])
