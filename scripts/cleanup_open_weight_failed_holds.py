#!/usr/bin/env python3
"""Clear finalized, excluded scheduler entries; never resume or replay them."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from audit_open_weight_trace import load_outcome,review_interception

ROOT=Path(__file__).resolve().parents[1]


def main():
    campaigns=sorted(ROOT.glob('runs/open_weight_timeline_main_20261102_v*'))
    constraint='('+' || '.join('Campaign == '+json.dumps(p.name) for p in campaigns)+') && JobStatus == 5'
    raw_ads=subprocess.check_output(['condor_q','-constraint',constraint,'-json'])
    ads=json.loads(raw_ads.strip() or b'[]')
    removed=[];pending=[]
    for ad in ads:
        args=shlex.split(ad['Args'])
        if '--task' not in args:continue
        campaign=Path(args[0]).parent;m=json.loads((campaign/'manifest.json').read_text())
        model=m['models'][int(args[args.index('--model')+1])];index=int(args[args.index('--task')+1])
        identifier=f"{ad['ClusterId']}.{ad['ProcId']}"
        outer=campaign/'models'/model['slug']/f'task{index:02d}'
        wrapper=load_outcome(campaign,outer,model)
        if wrapper is None:pending.append(identifier);continue
        path=campaign/wrapper['result'];raw=path.read_bytes();record=json.loads(raw);sample=record['samples'][0]
        complete=path.parent/'complete.json'
        if not complete.exists():pending.append(identifier);continue
        proof=json.loads(complete.read_text())
        if proof.get('attempt_id')!=sample['attempt_id'] or proof.get('result_sha256')!=hashlib.sha256(raw).hexdigest():
            pending.append(identifier);continue
        try:audit=review_interception(path,wrapper['audit'])
        except (AssertionError,ValueError,OSError) as error:audit={'passed':False,'error':type(error).__name__}
        if sample['infrastructure_status']=='valid' and not sample['monitor_errors'] and audit['passed']:
            continue  # The stricter valid-result cleanup handles these.
        subprocess.run(['condor_rm',identifier],check=True,capture_output=True)
        evidence={'time':datetime.now(timezone.utc).isoformat(),'job':identifier,'model':model['slug'],
            'task':m['tasks'][index]['task'],'attempt_id':sample['attempt_id'],'result_sha256':proof['result_sha256'],
            'complete_sha256':hashlib.sha256(complete.read_bytes()).hexdigest(),
            'native_status':sample['infrastructure_status'],'native_reason':sample.get('infrastructure_reason'),
            'trace_audit':audit,'action':'Remove finalized excluded scheduler hold. Original native artifacts unchanged. This script never replays a model.'}
        with (campaign/'reports/invalid_post_finalization_cleanup.jsonl').open('a') as f:f.write(json.dumps(evidence)+'\n')
        removed.append(identifier)
    print(json.dumps({'removed_finalized_excluded_holds':removed,'pending_unfinalized':pending}))


if __name__=='__main__':main()
