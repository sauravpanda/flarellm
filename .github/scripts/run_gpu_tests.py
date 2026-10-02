"""Run every ignored GPU test in its own process; retain crashes as failures."""
import json
import os
from pathlib import Path
import subprocess
import sys

output = Path(sys.argv[1] if len(sys.argv) > 1 else 'gpu-artifacts')
output.mkdir(parents=True, exist_ok=True)
report = {'passed': False, 'tests': []}

def save():
    (output / 'results.json').write_text(json.dumps(report, indent=2) + '\n')

try:
    build = subprocess.run([
        'cargo', 'test', '-p', 'flarellm-gpu', '--lib', '--tests', '--no-run',
        '--message-format=json',
    ], text=True, stdout=subprocess.PIPE)
    if build.returncode:
        print(build.stdout, flush=True)
        build.check_returncode()
    executables = set()
    for line in build.stdout.splitlines():
        item = json.loads(line)
        if item.get('executable') and item.get('profile', {}).get('test'):
            executables.add(item['executable'])
    for executable in sorted(executables):
        listing = subprocess.check_output([executable, '--ignored', '--list'], text=True)
        for line in listing.splitlines():
            if line.endswith(': test'):
                report['tests'].append({'executable': executable, 'name': line[:-6], 'status': 'not run'})
    if not report['tests']:
        raise RuntimeError('No ignored GPU tests discovered')
    save()
    environment = {**os.environ, 'FLARE_REQUIRE_GPU': '1'}
    for index, case in enumerate(report['tests']):
        print(f"\n=== {case['name']} ===", flush=True)
        try:
            result = subprocess.run([
                case['executable'], case['name'], '--exact', '--ignored',
                '--nocapture', '--test-threads=1',
            ], env=environment, text=True, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=120)
            case['returncode'] = result.returncode
            case['status'] = 'passed' if result.returncode == 0 else 'failed'
            log = result.stdout
        except subprocess.TimeoutExpired as error:
            case['status'] = 'failed'
            case['error'] = 'Timed out after 120 seconds'
            log = error.stdout or ''
            if isinstance(log, bytes):
                log = log.decode(errors='replace')
        case['log'] = f'test-{index:02d}.log'
        (output / case['log']).write_text(log)
        print(log, flush=True)
        print(f"{case['status']}: {case['name']} (exit {case.get('returncode', 'timeout')})", flush=True)
        save()
    report['passed'] = all(case['status'] == 'passed' for case in report['tests'])
except Exception as error:
    report['error'] = str(error)
finally:
    save()
    print(json.dumps(report, indent=2), flush=True)
sys.exit(0 if report['passed'] else 1)
