#!/usr/bin/env python3
"""Prepare a fresh snapshot for mechanical failures, never valid negatives."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

from audit_open_weight_trace import load_outcome, review_interception

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=ROOT/'runs/open_weight_timeline_main_20261102_v2')
    parser.add_argument('--retry-lock-failures',action='store_true',help='Permit one extra mechanical retry only for documented stale-budget-lock failures.')
    args=parser.parse_args();src=args.source.resolve();target=args.destination.absolute()
    dst=target.with_name('.'+target.name+'.preparing')
    assert not target.exists() and not dst.exists()
    m=json.loads((src/'manifest.json').read_text())
    indices=json.loads((src/'reports/task_selection.json').read_text())['indices']
    # One mechanical retry, with a bounded exception for the documented stale
    # mutex. Valid results are never replayed, whatever their success score.
    attempted_recoveries=Counter();lock_failures=set();valid_recoveries=set()
    for previous in ROOT.glob('runs/open_weight_timeline_main_20261102_v*'):
        if previous.resolve()==src or not (previous/'reports/recovery_selection.json').exists():continue
        previous_rows=json.loads((previous/'reports/recovery_selection.json').read_text())['failures']
        for row in previous_rows:
            identity=(row['model'],row['task_index']);attempted_recoveries[identity]+=1
            spec=next(x for x in m['models'] if x['slug']==row['model'])
            wrapper=load_outcome(previous,previous/'models'/row['model']/f"task{row['task_index']:02d}",spec)
            if wrapper is None:continue
            path=previous/wrapper['result'];s=json.loads(path.read_text())['samples'][0]
            if s['infrastructure_status']=='valid' and not s['monitor_errors'] and review_interception(path,wrapper['audit'])['passed']:
                valid_recoveries.add(identity)
            elif any('Global budget lock unavailable' in str(d.get('classifier_error','')) for d in s['decisions']):
                lock_failures.add(identity)
    failures=[]
    for model_index,model in enumerate(m['models']):
        for task_index in indices:
            identity=(model['slug'],task_index);prior_count=attempted_recoveries[identity]
            if identity in valid_recoveries:continue
            if prior_count and not (args.retry_lock_failures and prior_count==1 and identity in lock_failures):continue
            outer=src/'models'/model['slug']/f'task{task_index:02d}'
            d=load_outcome(src,outer,model)
            if d is None:continue
            path=src/d['result']
            sample=json.loads(path.read_text())['samples'][0]
            audit=review_interception(path,d['audit'])
            if sample['infrastructure_status']=='valid' and not sample['monitor_errors'] and audit['passed']:
                continue
            failures.append({'model_index':model_index,'model':model['slug'],'task_index':task_index,
                'task':m['tasks'][task_index]['task'],'original_result':str(path),
                'original_result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'infrastructure_reason':sample.get('infrastructure_reason'),
                'monitor_errors':sample['monitor_errors'],'trace_audit':audit})
            failures[-1]['prior_mechanical_recovery_count']=prior_count
            if prior_count:failures[-1]['additional_retry_reason']='The earlier mechanical recovery also hit the documented stale spending mutex; no valid result exists.'
    assert failures,'No mechanical failures to recover'
    dst.mkdir(parents=True)
    for name in ['source','assets']:shutil.copytree(src/name,dst/name)
    for name in ['worker.py','worker.sh']:shutil.copy2(src/name,dst/name)
    (dst/'reports').mkdir();(dst/'condor').mkdir()
    shutil.copy2(src/'reports/task_selection.json',dst/'reports/task_selection.json')
    shutil.copy2(src/'reports/offline_configuration_check.json',dst/'reports/offline_configuration_check.json')
    for model in m['models']:
        qualification=src/'models'/model['slug']/'qualification'
        shutil.copytree(qualification,dst/'models'/model['slug']/'qualification')
        catalog=dst/'assets'/Path(model['catalog_path']).name
        assert catalog.exists() and hashlib.sha256(catalog.read_bytes()).hexdigest()==model['catalog_sha256']
        model['catalog_path']=str(target/'assets'/catalog.name)
    m.update(campaign_id=target.name,created_at=datetime.now(timezone.utc).isoformat(),
             recovery_of=[src.name],parallel_limit=8,recovery_queue_limits={'qwen25':2,'other_models':6},
             purpose='Mechanical infrastructure-failure recovery only. Preserve task/model/monitor conditions and the shared USD130 main pool.',
             reasoning_effort_record='Explicit --reasoning-effort medium, identical to main v2.',
             fixture_seed_note='No explicit fixture seed or shared fixture bank was set. Original generators and defaults are preserved, including randomized ClawBench fixtures.',
             production_max_usd=130.0,
             qualification_reuse={'source':str(src),'reason':'Identical frozen source, catalogs, requested models and provider routes; no qualification calls repeated.'})
    (dst/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    (dst/'reports/recovery_selection.json').write_text(json.dumps({'selection_uses_success_outcome':False,'targeted_stale_lock_exception':args.retry_lock_failures,'failures':failures},indent=2)+'\n')
    rows=sorted(failures,key=lambda r:(r['task_index'],r['model_index']))
    (dst/'rows.txt').write_text(''.join(f"{r['model_index']} {r['task_index']}\n" for r in rows))
    submit=(src/'ready.sub').read_text().replace(str(src),str(target)).replace(src.name,target.name)
    submit=submit.replace('max_materialize = 30','max_materialize = 2').replace('max_idle = 30','max_idle = 2').replace('ready_rows.txt','rows.txt')
    (dst/'recovery.sub').write_text(submit)
    for name,selected,limit in [('qwen25',[r for r in rows if r['model']=='qwen25_72b'],2),
                                ('other',[r for r in rows if r['model']!='qwen25_72b'],6)]:
        (dst/(name+'_rows.txt')).write_text(''.join(f"{r['model_index']} {r['task_index']}\n" for r in selected))
        (dst/(name+'.sub')).write_text(submit.replace('from rows.txt','from '+name+'_rows.txt').replace('max_materialize = 2','max_materialize = '+str(limit)).replace('max_idle = 2','max_idle = '+str(limit)))
    dst.rename(target)
    print(json.dumps({'directory':str(target),'episodes':len(rows),'models':sorted({r['model'] for r in rows}),'submitted':False},indent=2))


if __name__=='__main__':main()
