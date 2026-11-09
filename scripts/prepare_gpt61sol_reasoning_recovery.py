"""Recover infrastructure-invalid first-stop episodes without replaying valid ones."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/gpt61sol_reasoning_no_continuation_20261107_v3'
DEST = ROOT / 'runs/gpt61sol_reasoning_no_continuation_recovery_20261107_v1'

def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def write(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def main():
    assert not DEST.exists()
    manifest = json.loads((OLD / 'manifest.json').read_text())
    outcomes = [json.loads(p.read_text()) for p in OLD.glob('episodes/*/outcome.json')]
    assert len(outcomes) == 150
    failed = sorted([r for r in outcomes if not r['valid']], key=lambda r: r['row']['index'])
    assert len(failed) == 7
    for name in ['assets', 'source', 'qualification', 'condor', 'reports', 'episodes']:
        (DEST / name).mkdir(parents=True)
    shutil.copytree(OLD / 'source', DEST / 'source', dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(OLD / 'assets', DEST / 'assets', dirs_exist_ok=True)
    shutil.copytree(OLD / 'qualification', DEST / 'qualification', dirs_exist_ok=True)
    for name in ['worker.py', 'worker.sh', 'audit_open_weight_trace.py', 'audit_codex_code_mode.py']:
        shutil.copy2(OLD / name, DEST / name)
    shutil.copy2(Path(__file__), DEST / 'prepare.py')
    budget = DEST / 'source/src/instrumental_evasion/runner/codex_budget.py'
    text = budget.read_text()
    substitutions = {
        'REVISION = "response_completion_hook_latency_v3"': 'REVISION = "response_completion_code_mode_v4"',
        '        self.session_tool_calls = 0\n': '        self.session_tool_calls = 0\n        self.code_mode_seen = False\n',
        '            if self.stdout_tool_starts > self.session_tool_calls:':
            '            # One code-mode response can dispatch many native tools. Its\n'
            '            # response-item count is not comparable to stdout tool starts.\n'
            '            # Session existence, per-response usage and monotonic counters\n'
            '            # remain mandatory; execution binding is audited separately.\n'
            '            if not self.code_mode_seen and self.stdout_tool_starts > self.session_tool_calls:',
        '                "session_tool_calls": self.session_tool_calls}':
            '                "session_tool_calls": self.session_tool_calls,\n'
            '                "tool_count_parity_applicable": not self.code_mode_seen,\n'
            '                "code_mode_seen": self.code_mode_seen}',
        '                    if record.get("type") == "response_item" and isinstance(payload, dict):\n':
            '                    if record.get("type") == "response_item" and isinstance(payload, dict):\n'
            '                        if (payload.get("type") == "custom_tool_call"\n'
            '                                and payload.get("namespace") == "functions"\n'
            '                                and payload.get("name") == "exec"):\n'
            '                            self.code_mode_seen = True\n',
    }
    for original, replacement in substitutions.items():
        assert text.count(original) == 1, original
        text = text.replace(original, replacement)
    budget.write_text(text)
    sys.path.insert(0, str(DEST / 'source/src'))
    from instrumental_evasion.runner.attempt_artifact import new_attempt_id
    rows = []
    for index, outcome in enumerate(failed):
        row = dict(outcome['row'])
        row.update(index=index, original_index=outcome['row']['index'], attempt_id=new_attempt_id(),
                   previous_attempt_id=outcome['row']['attempt_id'], previous_result=outcome['result'],
                   previous_result_sha256=outcome['result_sha256'], retry_reason=outcome['error'])
        rows.append(row)
    manifest.update(campaign_id=DEST.name, created_at=datetime.now(timezone.utc).isoformat(),
                    parent_campaign=str(OLD), parent_manifest_sha256=sha(OLD / 'manifest.json'),
                    rows=rows, planned_episodes=7, task_count=len({r['task'] for r in rows}),
                    maximum_concurrency=7, qualification_reuse=str(OLD / 'qualification'),
                    valid_original_episodes_preserved=143,
                    source_sha256={str(p.relative_to(DEST / 'source')): sha(p) for p in sorted((DEST / 'source').rglob('*')) if p.is_file()})
    manifest['environment']['CLAW_FIXTURE_BANK_PATH'] = str(DEST / 'assets/fixture-bank-epoch0.json')
    manifest['notes'] += ['Recovery only: six interrupted by a false code-mode tool-count comparison; one failed hook probe.',
                          'One functions.exec proposal may execute multiple native tools; response-item count and stdout tool-start count are not comparable. Disable only this parity check in observed code-mode sessions.',
                          'All session presence, response usage, counter validation and token-limit checks are retained. Full native tool execution audit remains separate.',
                          'Fresh attempts on the same fixtures and settings; no successful infrastructure-valid outcome is rerun.',
                          '16 GB job memory and exclusion of g158/g166 address cluster holds; model parameters are unchanged.']
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
requirements = !regexp("^g(089|132|158|166)", TARGET.Machine, "i")
request_cpus = 2
request_memory = 16384
request_disk = 40G
on_exit_remove = True
kill_sig = SIGTERM
kill_sig_timeout = 600
'''
    (DEST / 'episode.sub').write_text(common + f'''arguments = {DEST}/worker.sh --row $(row)
output = condor/episode.$(Cluster).out
error = condor/episode.$(Cluster).err
log = condor/episode.$(Cluster).log
queue
''')
    dag = []
    for row in rows:
        node = 'RETRY_' + str(row['index'])
        dag += [f'JOB {node} {DEST}/episode.sub', f'VARS {node} row="{row["index"]}"']
    (DEST / 'main.dag').write_text('\n'.join(dag) + '\n')
    (DEST / 'workflow.config').write_text('DAGMAN_CONDOR_SUBMIT_EXE = /usr/bin/condor_submit\n')
    write(DEST / 'launcher-checksums.json', {p.name: sha(p) for p in DEST.iterdir() if p.is_file()})
    print(json.dumps({'campaign': str(DEST), 'rows': [(r['task'], r['effort']) for r in rows]}))

if __name__ == '__main__':
    main()
