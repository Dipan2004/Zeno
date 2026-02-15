#!/usr/bin/env python3
import subprocess
import sys

# Run test_phase5.py with 'n' input for voice test
result = subprocess.run(
    [sys.executable, 'test_phase5.py'],
    input='n\n',
    text=True,
    timeout=45
)

sys.exit(result.returncode)
