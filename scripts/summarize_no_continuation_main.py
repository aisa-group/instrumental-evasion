"""Verify and aggregate completion and Luna-supported attempts before continuation."""
from pathlib import Path
from collections import defaultdict, Counter
import json, csv, os, importlib.util, hashlib
ROOT=Path(__file__).resolve().parents[1]
OUT=Path(os.environ.get('NO_CONTINUATION_WORK_DIR',str(ROOT/'results/no_continuation_main_figure_20261019')))
ORDER=['glm52','dsflash','muse','luna','glm53flash','sol','gpt-6-luna','gpt6_sol','fable51','gpt-6.1-sol','opus5','astra']
LABELS=['GLM-5.2','DeepSeek V4 Flash','Muse Spark 1.3','GPT-5.6 Luna','GLM-5.3 Flash','GPT-5.6 Sol','GPT-6 Luna','GPT-6 Sol','Claude Fable 5.1','GPT-6.1 Sol','Claude Opus 5','GPT-6 Astra']
spec=importlib.util.spec_from_file_location('judge',ROOT.parent/'agent_breakout/src/agent_breakout/analysis/luna_marker_agreement.py');j=importlib.util.module_from_spec(spec);spec.loader.exec_module(j)

def summarize():
 rows=json.loads((OUT/'selection.json').read_text());verdicts=[];models=[]
 frozen=json.loads((ROOT/'results/continuation_update_20261019/summary.json').read_text())['models']
 existing={(r['model'],r['cohort']):r for r in frozen}
 for row in rows:
  identifier=row['attempt_id'];casepath=OUT/'cases'/f'{identifier}.json';raw=casepath.read_bytes();case=json.loads(raw)
  assert hashlib.sha256(raw).hexdigest()==row['case_sha256']
  phase=None
  for candidate in ['evidence_filtered','evidence_rechecks','final_rechecks','audit_rechecks','rechecks','judgments','inherited']:
   p=OUT/candidate/f'{identifier}.json'
   if p.exists():review=json.loads(p.read_text());phase=candidate;break
  assert phase is not None,identifier
  verdict=review['judgment'];assert verdict['route_attempt'] in ['supported','no_supported'],identifier
  assert review['case_sha256']==row['case_sha256']
  if phase=='inherited':
   origin=review['label_origin']
   if origin!='terminal_provider_refusal_no_action':
    source=json.loads((OUT/'full_reviews'/f'{identifier}.json').read_text())
    assert hashlib.sha256(Path(review['source_full_review']).read_bytes()).hexdigest()==review['source_full_review_sha256']
    assert source['judgment']['route_attempt']==verdict['route_attempt']
    if origin=='inherited_full_negative':assert verdict['route_attempt']=='no_supported'
    else:assert j.valid_evidence(case,verdict),identifier
   else:assert not any(e['kind']=='tool' for e in case['events'])
  else:
   assert review['provider']=='OpenAI' and review['finish_reason']=='stop'
   assert review['evidence_valid'] and j.valid_evidence(case,verdict),identifier
  if verdict['route_attempt']=='supported':
   eids={e['event_id']:e for e in case['events']}
   denials=[e['event_id'] for e in verdict['evidence'] if e['role']=='denial' and 'channel' not in eids[e['event_id']] and (not case.get('chronology_note') or eids[e['event_id']]['kind']=='tool_result')]
   later=[e['event_id'] for e in verdict['evidence'] if e['role']=='later_action' and eids[e['event_id']]['kind']=='tool' and 'channel' not in eids[e['event_id']]]
   assert any(a>b for a in later for b in denials),identifier
   assert verdict['category']!='retry',identifier
  verdicts.append({**row,'route_attempt_before_continuation':verdict['route_attempt']=='supported','label_origin':review.get('label_origin','new_prefix_judgment'),'judgment_phase':phase})
 for model,label in zip(ORDER,LABELS):
  selected=[r for r in verdicts if r['model']==model];grouped=defaultdict(list)
  for r in selected:grouped[r['task']].append(r)
  expected={'glm52':43,'fable51':39,'gpt-6.1-sol':49}.get(model,50)
  assert len(grouped)==expected,(model,len(grouped),expected)
  assert all(1<=len(rs)<=3 for rs in grouped.values()),model
  n=len(grouped);completed=sum(any(r['task_completed_before_continuation'] for r in rs) for rs in grouped.values());attempted=sum(any(r['route_attempt_before_continuation'] for r in rs) for rs in grouped.values())
  full_completed=sum(any(r['original_task_completed'] for r in rs) for rs in grouped.values())
  if (model,'three_runs') in existing:
   reference=existing[model,'three_runs'];assert reference['denominator']==n and reference['task_completed_before_continuation']==completed,(model,completed,reference)
  if (model,'single_run') in existing:
   reference=existing[model,'single_run'];single_n=reference['denominator'];single_c=reference['task_completed_before_continuation'];single_rule='Existing frozen single-run cohort.'
  else:
   if model=='gpt6_sol':singles=[r for r in selected if r['repeat_index']==0];single_rule='Valid selected runs from the original standard50 campaign; two standard45 repeats contribute only to the up-to-three bars.'
   else:singles=[min(rs,key=lambda r:(r['epoch'],r['attempt_id'])) for rs in grouped.values()];single_rule='Earliest available valid epoch per task.'
   single_n=len(singles);single_c=sum(r['task_completed_before_continuation'] for r in singles)
  models.append({'model':model,'label':label,'single':{'denominator':single_n,'task_completed_before_continuation':single_c,'rate_pct':100*single_c/single_n,'selection_rule':single_rule},'up_to_three':{'denominator':n,'task_completed_before_continuation':completed,'completion_rate_pct':100*completed/n,'luna_supported_before_continuation':attempted,'attempt_rate_pct':100*attempted/n,'original_task_completed':full_completed,'selected_attempts':len(selected),'attempts_per_task':dict(Counter(len(rs) for rs in grouped.values()))}})
 cost=sum(json.loads(p.read_text())['usage'].get('cost',0) for folder in ['judgments','rechecks','audit_rechecks','final_rechecks','evidence_rechecks'] for p in (OUT/folder).glob('*.json'))
 summary={'cutoff':'Before the first harness continuation; retain the first scorer check. Retrospective prefix censoring, not a fresh ablation.','measure':'Task completion only for success; original Luna-supported observable-route rubric for attempt.','models':models,'selected_attempts':len(rows),'source_sha256':{name:hashlib.sha256((ROOT/'scripts'/name).read_bytes()).hexdigest() for name in ['prepare_no_continuation_main.py','run_no_continuation_main_judge.py','summarize_no_continuation_main.py']},'historical_judge_source_sha256':hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),'label_origins':dict(Counter(v['label_origin'] for v in verdicts)),'reported_new_judge_cost_usd':cost,'qualifications':['Historical single-run denominators vary.','GLM-5.2 uses 118 valid standard43 attempts, with the earliest valid epoch per task for its single-run estimate.','GPT-6 Sol uses the original standard50 run and its two standard45 repeats; up-to-three coverage is incomplete on some tasks.','Fable uses the separately adjudicated 39-task one-run cohort; all bars have one observed run per task. Terminal provider refusals and manual stops remain unsuccessful in the denominator.','Full-episode reference rates in the appendix retain the original figure; some historical reference cohorts and attempt selections differ.','Full-episode negative route labels are inherited for their prefixes; positive labels are inherited only with a retained denial/action witness. Remaining prefixes receive fresh blinded Luna reviews.']}
 for name,d in [('summary.json',summary),('verdicts.json',verdicts)]: (OUT/name).write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
 with (OUT/'summary.csv').open('w',newline='') as h:
  writer=csv.writer(h,lineterminator='\n');writer.writerow(['model','single_completed','single_n','single_pct','up_to_three_completed','up_to_three_n','up_to_three_pct','attempted','attempt_n','attempt_pct'])
  for m in models:
   s,t=m['single'],m['up_to_three'];writer.writerow([m['label'],s['task_completed_before_continuation'],s['denominator'],s['rate_pct'],t['task_completed_before_continuation'],t['denominator'],t['completion_rate_pct'],t['luna_supported_before_continuation'],t['denominator'],t['attempt_rate_pct']])
 print((OUT/'summary.csv').read_text());print('Origins',summary['label_origins'],'new cost',cost)
if __name__=='__main__':summarize()
