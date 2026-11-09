#!/usr/bin/env python3
"""Remove scheduler holds only after independently verified native finalization."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from audit_open_weight_trace import load_outcome, review_interception

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
        campaign=Path(args[0]).parent
        m=json.loads((campaign/'manifest.json').read_text());model=m['models'][int(args[args.index('--model')+1])];i=int(args[args.index('--task')+1])
        outer=campaign/'models'/model['slug']/f'task{i:02d}';outcome=outer/'outcome.json'
        identifier=f"{ad['ClusterId']}.{ad['ProcId']}"
        native=load_outcome(campaign,outer,model)
        if native is None:pending.append({'job':identifier,'reason':'No finalized native outcome'});continue
        try:
            path=campaign/native['result'];record=json.loads(path.read_text());sample=record['samples'][0]
            audit=review_interception(path,native['audit'])
            assert audit['passed'] and audit['observed_models']==[model['model']]
            assert sample['infrastructure_status']=='valid' and not sample['monitor_errors']
            assert record['metadata']['monitor_history'] is True and record['metadata']['monitor_history_max_calls']==40
            assert record['metadata']['policy_prompt_sha256']==m['tasks'][i]['policy_sha256']
            for item in sample['metadata']['codex_session_export']['files']:
                if item.get('status')=='collected':
                    assert not item.get('export_kind')
                    assert hashlib.sha256((path.parent/item['path']).read_bytes()).hexdigest()==item['sha256']
        except (AssertionError,KeyError,ValueError,OSError) as error:
            pending.append({'job':identifier,'reason':'Native verification failed: '+type(error).__name__});continue
        result=subprocess.run(['condor_rm',identifier],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        row={'time':datetime.now(timezone.utc).isoformat(),'job':identifier,'model':model['slug'],'task':m['tasks'][i]['task'],
             'attempt_id':sample['attempt_id'],'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
             'native_stop_reason':sample['stop_reason'],'scheduler_hold_reason':ad.get('HoldReason'),
             'action':'Remove scheduler entry after verified native finalization; no replay.'}
        with (campaign/'reports/post_finalization_cleanup.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
        removed.append(row)
    print(json.dumps({'removed_finalized_holds':len(removed),'pending':pending},indent=2))


if __name__=='__main__':main()
