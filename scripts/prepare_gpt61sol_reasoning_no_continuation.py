"""Freeze the matched 30-task, five-effort, first-invocation-only campaign."""
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import shutil
import httpx

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "runs/gpt61sol_reasoning_no_continuation_20261107_v3"
OLD = Path('/lustre/fast/fast/dschmotz/agent_breakout/results/experiment_tracker/reasoning_effort_gpt6_sol_unchanged_claw26_20260922_v1')
PREVIOUS = ROOT / 'runs/gpt61sol_gpt6luna_standard50_subscription_20261016_v1'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def main():
    assert not DEST.exists(), 'Never overwrite an existing campaign'
    old = json.loads((OLD / 'protocol.json').read_text())
    prior = json.loads((PREVIOUS / 'manifest.json').read_text())
    scans = json.loads((ROOT / 'results/figure4_no_continuation_20261107_v1/scans/gpt6sol/episodes.json').read_text())
    baseline = {r['task']: r for r in scans if r['effort'] == 'low'}
    assert len(baseline) == 30
    for name in ['assets', 'source', 'condor', 'reports', 'qualification', 'episodes']:
        (DEST / name).mkdir(parents=True)
    shutil.copytree(ROOT / 'src', DEST / 'source/src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    # The same session telemetry schema is emitted by the installed 0.163.0.
    # Keep this compatibility change isolated to this campaign's source snapshot.
    codex = DEST / 'source/src/instrumental_evasion/runner/scaffolds/codex.py'
    source = codex.read_text()
    guard = 'or cli_version(self.binary) != "codex-cli 0.153.3"'
    assert source.count(guard) == 1
    source = source.replace(guard, 'or cli_version(self.binary) not in {"codex-cli 0.153.3", "codex-cli 0.163.0"}')
    source = source.replace('requires local Codex 0.153.3 with API usage.', 'requires local Codex 0.153.3 or 0.163.0 with API usage.')
    codex.write_text(source)
    for name in ['AGENTS.md', 'pyproject.toml', 'uv.lock']:
        shutil.copy2(ROOT / name, DEST / 'source' / name)
    shutil.copy2(PREVIOUS / 'assets/python_packages.tar', DEST / 'assets/python_packages.tar')
    shutil.copy2(OLD / 'fixture-bank-epoch0.json', DEST / 'assets/fixture-bank-epoch0.json')
    shutil.copy2(ROOT / 'scripts/gpt61sol_reasoning_worker.py', DEST / 'worker.py')
    shutil.copy2(Path(__file__), DEST / 'prepare.py')
    shutil.copy2(ROOT / 'scripts/audit_open_weight_trace.py', DEST / 'audit_open_weight_trace.py')
    shutil.copy2(ROOT / 'scripts/audit_codex_code_mode.py', DEST / 'audit_codex_code_mode.py')
    shutil.copy2(PREVIOUS / 'worker.sh', DEST / 'worker.sh')
    env = dict(old['environment'])
    env.update(AGENT_MAX_TURNS='1', AGENT_ATTEMPTS='1', CODEX_OPENROUTER_PROVIDER_PIN='azure',
               CODEX_TOOL_INTERFACE='default', MONITOR_HISTORY='1', MONITOR_HISTORY_MAX_CALLS='40',
               MONITOR_HISTORY_MAX_ARG_CHARS='600', MONITOR_HISTORY_OUTCOMES='1',
               AUTHORED10_FIXTURE_SEED=old['fixture_seed'],
               CLAW_FIXTURE_BANK_PATH=str(DEST / 'assets/fixture-bank-epoch0.json'),
               IE_IMAGE_DIR=prior['image_directory'], CODEX_BIN=prior['codex_binary'])
    os.environ.update(env)
    from instrumental_evasion.tasks.registry import resolve
    from instrumental_evasion.policies import policy_prompt_sha256, trimmed_policy_id
    from instrumental_evasion.runner.attempt_artifact import new_attempt_id
    from instrumental_evasion.tasks.clawbench.specs import validate_configured_fixture_bank
    validate_configured_fixture_bank(list(baseline))
    references, images = {}, {}
    for name, row in sorted(baseline.items()):
        result = json.loads(Path(row['result']['path']).read_text())
        task = resolve(name, 'monitored')[0]
        prompt = next(m['content'] for m in result['samples'][0]['messages'] if m['role'] == 'user')
        assert prompt == task.spec.prompt, name
        policy_hash = policy_prompt_sha256(trimmed_policy_id(task.policy), history_disclosed=True)
        assert policy_hash == result['metadata']['policy_prompt_sha256'], name
        image = task.spec.image()
        if image not in images:
            images[image] = sha(image)
        expected = old['images'].get(name, old['images']['claw_file_001_csv_markdown'])['sha256']
        assert images[image] == expected, (name, 'Historical image changed')
        references[name] = {'historical_result': row['result'], 'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(),
                            'policy_prompt_sha256': policy_hash, 'image': image, 'image_sha256': images[image],
                            'historical_fixture_id': result['samples'][0]['metadata'].get('fixture_id')}
    with httpx.Client(timeout=40) as client:
        response = client.get('https://openrouter.ai/api/v1/models/openai/gpt-6.1-sol/endpoints')
        response.raise_for_status()
        endpoints = response.json()
        write(DEST / 'assets/endpoints-gpt61sol.json', endpoints)
        matches = [e for e in endpoints['data']['endpoints'] if e.get('tag') == 'azure' and e.get('status') == 0]
        assert len(matches) == 1
    efforts = ['low', 'medium', 'high', 'xhigh', 'max']
    rows = [{'index': i, 'task': name, 'effort': effort, 'epoch': 0, 'attempt_id': new_attempt_id()}
            for i, (name, effort) in enumerate((name, effort) for name in sorted(baseline) for effort in efforts)]
    manifest = {'campaign_id': DEST.name, 'created_at': datetime.now(timezone.utc).isoformat(),
                'model': 'openrouter/openai/gpt-6.1-sol', 'requested_model_id': 'gpt-6.1-sol',
                'recorded_model': 'openai/gpt-6.1-sol', 'provider_only': ['azure'], 'allow_fallbacks': False,
                'accepted_served_model_ids': ['openai/gpt-6.1-sol', 'openai/gpt-6.1-sol-20260929'],
                'auth': 'api_key', 'credential_file': prior['monitor_credential_file'], 'credential_retained': False,
                'scaffold': 'codex', 'runtime': 'apptainer', 'codex_version': '0.163.0',
                'codex_binary_sha256': sha(prior['codex_binary']), 'environment': env,
                'monitor_model': 'openrouter/openai/gpt-oss-120b', 'monitor_settings': {k:v for k,v in env.items() if k.startswith('MONITOR_')},
                'no_continuations': True, 'continuation_condition': 'none', 'continuation_message': None,
                'limits': {'max_turns': 1, 'attempts': 1, 'time_limit_seconds': 3600, 'token_limit': 20000000},
                'efforts': efforts, 'task_count': 30, 'planned_episodes': 150, 'rows': rows,
                'fixture_seed': old['fixture_seed'], 'fixture_bank_sha256': sha(DEST / 'assets/fixture-bank-epoch0.json'),
                'qualification_reuse': str(ROOT / 'runs/gpt61sol_reasoning_no_continuation_20261107_v2/qualification'),
                'references': references, 'maximum_concurrency': 20, 'automatic_retries': False,
                'source_sha256': {str(p.relative_to(DEST / 'source')): sha(p) for p in sorted((DEST / 'source').rglob('*')) if p.is_file()},
                'dependency_archive_sha256': sha(DEST / 'assets/python_packages.tar'),
                'notes': ['Matched historical GPT-6 Sol 30-task cohort; one independent session per task and effort.',
                          'Supersedes v1 qualification-only failure: an obsolete watchdog version guard rejected 0.163.0 before API calls.',
                          'Campaign-local source guard accepts 0.163.0; unchanged watchdog parses its original usage records. Live telemetry required in qualification.',
                          'Re-audits retained v2 qualification sessions using exact nested execution IDs; does not replay valid qualification outcomes.',
                          'Exactly one CLI invocation per episode; no continuation or goal prompt.',
                          'CLI 0.163.0 and current frozen scaffold differ from historical CLI 0.153.3.',
                          'Full original session JSONL required, including generated encrypted reasoning.',
                          'Task completion and benchmark completion-or-effect scores are reported separately.',
                          'Trace/interception and infrastructure failures remain visible; valid failures are never replayed.']}
    write(DEST / 'manifest.json', manifest)
    common = f'''executable = /bin/bash
initialdir = {DEST}
getenv = HOME,USER,PATH,http_proxy,https_proxy,HTTP_PROXY,HTTPS_PROXY
+BypassLXCfs = "true"
+MaxRuntime = 5400
+Campaign = "{DEST.name}"
+bid = 15
priority = -985
JobBatchName = "{DEST.name}"
requirements = !regexp("^g(089|132)", TARGET.Machine, "i")
request_cpus = 2
request_memory = 8192
request_disk = 40G
on_exit_remove = True
kill_sig = SIGTERM
kill_sig_timeout = 600
'''
    for kind, args in [('qualification', '--qualify $(effort)'), ('episode', '--row $(row)'), ('report', '--report')]:
        (DEST / (kind + '.sub')).write_text(common + f'''arguments = {DEST}/worker.sh {args}
output = condor/{kind}.$(Cluster).out
error = condor/{kind}.$(Cluster).err
log = condor/{kind}.$(Cluster).log
queue
''')
    dag = []
    for effort in efforts:
        node = 'QUAL_' + effort
        dag.extend([f'JOB {node} {DEST}/qualification.sub', f'VARS {node} effort="{effort}"'])
    for row in rows:
        node = 'EP_' + str(row['index'])
        dag.extend([f'JOB {node} {DEST}/episode.sub', f'VARS {node} row="{row["index"]}"',
                    f'CATEGORY {node} episodes', f'PARENT QUAL_{row["effort"]} CHILD {node}'])
    dag.extend(['MAXJOBS episodes 20', f'FINAL REPORT {DEST}/report.sub'])
    (DEST / 'main.dag').write_text('\n'.join(dag) + '\n')
    (DEST / 'workflow.config').write_text('DAGMAN_CONDOR_SUBMIT_EXE = /usr/bin/condor_submit\n')
    write(DEST / 'launcher-checksums.json', {p.name: sha(p) for p in DEST.iterdir() if p.is_file()})
    print(json.dumps({'campaign': str(DEST), 'episodes': len(rows), 'images_verified': len(images)}))

if __name__ == '__main__':
    main()
