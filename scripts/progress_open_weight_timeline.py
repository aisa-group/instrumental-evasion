#!/usr/bin/env python3
"""Read-only coverage and spending report for the release-date experiment."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_open_weight_trace import review_interception
from summarize_open_weight_timeline import LABELS

ROOT=Path(__file__).resolve().parents[1]
models={m:{} for m in LABELS}
attempted=Counter();invalid=Counter();cost=0.;upper=0.;seen_qual=set()
for v in [3,4,5,6,7]:
    root=ROOT/f'runs/open_weight_timeline_20261102_v{v}'
    for p in root.glob('models/**/spending.json'):
        if p.parent.name=='qualification':
            record=p.parent/'record.json'
            identity=hashlib.sha256(record.read_bytes()).hexdigest() if record.exists() else str(p)
            if identity in seen_qual:continue
            seen_qual.add(identity)
        d=json.loads(p.read_text());upper+=d['used_upper_usd']
        cost+=sum(float(e.get('reported_cost_usd') or 0) for e in d['events'] if e['event']=='settle')
    for slug in LABELS:
        for p in (root/'models'/slug).glob('task*/outcome.json'):
            attempted[slug]+=1;o=json.loads(p.read_text());result=root/o['result'];s=json.loads(result.read_text())['samples'][0]
            if s['infrastructure_status']=='valid' and not s['monitor_errors'] and review_interception(result,o['audit'])['passed']:
                models[slug].setdefault(p.parent.name,s)
            else:invalid[slug]+=1
common=set.intersection(*(set(v) for v in models.values()))
report={'models':{m:{'finalized_attempts':attempted[m],'valid_tasks':len(rr),'invalid_attempts':invalid[m],
                     'task_completions':sum(s['metadata'].get('success_scoring',{}).get('task_completed',s['solved']) for s in rr.values())} for m,rr in models.items()},
        'fully_matched_valid_tasks':len(common),'reported_api_cost_usd':round(cost,4),'spending_upper_usd_including_inflight_or_uncertain':round(upper,4)}
print(json.dumps(report,indent=2))
