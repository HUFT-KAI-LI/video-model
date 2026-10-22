"""Reuse evidence archiving and publishing; include the independent control tests."""
import os
from pathlib import Path
import subprocess
import sys

from ..wan_said.io_utils import sha256, write_json
from ..wan_said.pilot_v2.paired_eval.audit import publish, snapshot


def archive(run, output, upload=False):
    output = Path(output)
    snapshot(run, output)
    here = Path(__file__).resolve().parent
    with (output/'fm_control_tests.log').open('w') as log:
        result = subprocess.run([sys.executable, '-m', 'unittest', 'discover', '-s', str(here/'tests'), '-v'],
            stdout=log, stderr=subprocess.STDOUT, env=dict(os.environ))
    write_json(output/'fm_control_source_sha256.json', {str(p.relative_to(here)): sha256(p)
        for p in here.rglob('*.py') if '__pycache__' not in p.parts})
    write_json(output/'artifact_sha256.json', {str(p.relative_to(output)): sha256(p)
        for p in output.rglob('*') if p.is_file() and p != output/'artifact_sha256.json'})
    if result.returncode:
        raise RuntimeError('FM control tests failed; audit was preserved but not published')
    return publish(output) if upload else None
