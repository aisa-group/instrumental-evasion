#!/usr/bin/env python3
"""Reconcile captured response IDs with OpenRouter's serving metadata."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import re

import httpx


def normalized(value):
    return re.sub('[^a-z0-9]','',(value or '').lower())


def main():
    p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('--qualification-only',action='store_true');p.add_argument('--workers',type=int,default=2);args=p.parse_args()
    m=json.loads((args.campaign/'manifest.json').read_text());key=Path(m['credential_file']).read_text().strip()
    aliases_path=args.campaign/'assets/requested_model_aliases.json'
    if aliases_path.exists():
        catalog=json.loads(aliases_path.read_text())['models']
    else:
        catalog=httpx.get('https://openrouter.ai/api/v1/models',timeout=30).json()['data']
        catalog=[{k:r.get(k) for k in ['id','canonical_slug','name','created','hugging_face_id']} for r in catalog if r['id'] in {v['model'] for v in m['models']}]
        aliases_path.write_text(json.dumps({'source':'https://openrouter.ai/api/v1/models','models':catalog},indent=2)+'\n')
    aliases={r['id']:{r['id'],r.get('canonical_slug') or r['id']} for r in catalog}
    out=args.campaign/'reports/serving_metadata';out.mkdir(parents=True,exist_ok=True)
    expected={}
    for path in args.campaign.glob('models/**/openrouter-gateway.jsonl'):
        if args.qualification_only and path.parent.name!='qualification':continue
        for line in path.read_text().splitlines():
            try:e=json.loads(line)
            except ValueError:continue
            if e.get('response_id'):
                expected[e['response_id']]={'model':e['model'],'provider':e['provider_only'][0]}
    def lookup(identifier):
        path=out/(identifier+'.json')
        if path.exists():
            d=json.loads(path.read_text())
            if d.get('lookup_status')==200:return identifier,d
        r=httpx.get('https://openrouter.ai/api/v1/generation',params={'id':identifier},headers={'Authorization':'Bearer '+key},timeout=30)
        data=r.json().get('data',{}) if r.status_code==200 else {}
        d={'lookup_status':r.status_code,**{k:data.get(k) for k in ['id','model','provider_name','total_cost','native_tokens_prompt','native_tokens_completion','native_tokens_reasoning']}}
        path.write_text(json.dumps(d,indent=2)+'\n');return identifier,d
    mismatches=[];pending=[];cost=0.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures={pool.submit(lookup,k):k for k in expected}
        for f in as_completed(futures):
            identifier=futures[f]
            try:identifier,data=f.result()
            except Exception as error:pending.append({'id':identifier,'error':type(error).__name__});continue
            if data['lookup_status']!=200:pending.append({'id':identifier,'status':data['lookup_status']});continue
            exp=expected[identifier]
            if data['model'] not in aliases.get(exp['model'],{exp['model']}) or normalized(data['provider_name'])!=normalized(exp['provider']):
                mismatches.append({'id':identifier,'expected':exp,'served':data})
            cost+=float(data.get('total_cost') or 0)
    report={'responses':len(expected),'passed':bool(expected and not mismatches and not pending),'mismatches':mismatches,'pending':pending,
            'audit_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'generation_lookup_cost_usd':cost,'accepted_model_identifiers':{k:sorted(v) for k,v in aliases.items()},
            'note':'Model equivalence uses only the broker catalog canonical_slug mapping; requested IDs remain unchanged. Serving metadata is not a cryptographic weight fingerprint. Charges are calculated from retained response usage, since these generation lookups can omit usage.'}
    (args.campaign/'reports/serving_route_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in {'mismatches','pending','accepted_model_identifiers'}},indent=2))


if __name__=='__main__':main()
