"""Remove held scheduler entries only after every native result is finalized."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def main():
    raw = subprocess.check_output(['condor_q', '-constraint',
        'regexp("^open_weight_extension_", Campaign) && JobStatus == 5', '-json'])
    rows = [];pending = []
    for ad in json.loads(raw.strip() or b'[]'):
        args = shlex.split(ad['Args']);root = Path(args[0]).parent
        if '--qualify' in args or '--references' in args:continue
        m = json.loads((root/'manifest.json').read_text());mi = int(args[args.index('--model')+1])
        base = root/'models'/m['models'][mi]['slug']
        identifier = f"{ad['ClusterId']}.{ad['ProcId']}"
        if '--task' in args:
            base = base / f"task{int(args[args.index('--task')+1]):02d}";expected = 1
        else:
            shard = int(args[args.index('--shard')+1]);base = base/f'shard{shard}';expected = len(m['task_shards'][shard])
        paths = list(base.glob('*/epoch0/*/result.json'))
        verified = []
        for path in paths:
            proof = path.parent/'complete.json'
            if not proof.exists():continue
            r = json.loads(path.read_text());p = json.loads(proof.read_text());s = r['samples'][0]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if p.get('attempt_id') == s['attempt_id'] and p.get('result_sha256') == digest:
                verified.append({'path':str(path), 'sha256':digest, 'attempt_id':s['attempt_id'], 'status':s['infrastructure_status']})
        if len(paths) != expected or len(verified) != expected:
            pending.append({'job':identifier, 'finalized':len(verified), 'expected':expected});continue
        subprocess.run(['condor_rm', identifier], check=True, capture_output=True)
        row = {'time':datetime.now(timezone.utc).isoformat(), 'job':identifier, 'hold_reason':ad.get('HoldReason'),
               'results':verified, 'action':'Remove scheduler entry after native finalization; no replay or result edits.'}
        with (root/'reports/post_finalization_cleanup.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        rows.append(identifier)
    print(json.dumps({'removed':rows, 'pending':pending}))


if __name__ == '__main__': main()
