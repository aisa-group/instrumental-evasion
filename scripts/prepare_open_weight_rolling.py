"""Freeze a scheduling-only restart without resetting spending or observations."""
import hashlib
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / 'runs/open_weight_high_control_20261103_v1'
NEW = ROOT / 'runs/open_weight_high_control_20261103_v2'
OLD_OUT = ROOT / 'results/open_weight_release_dates_high_20261103_v1'
OUT = ROOT / 'results/open_weight_release_dates_high_20261103_v2'


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def replace(text, old, new):
    assert old in text, old
    return text.replace(old, new)


def main():
    resume = '--resume' in sys.argv
    assert not (NEW / 'reports/restart_manifest.json').exists(), 'Never overwrite a prepared controller'
    assert resume or not (NEW / 'manifest.json').exists(), 'Use --resume for interrupted preparation only'
    NEW.mkdir(exist_ok=True)
    (NEW / 'reports').mkdir(exist_ok=True)
    (NEW / 'condor').mkdir(exist_ok=True)
    for name in ['spending_authorization.json', 'budget_seed.json', 'task_selection.csv']:
        shutil.copyfile(OLD / name, NEW / name)
    for folder in ['assets']:
        shutil.copytree(OLD / folder, NEW / folder, dirs_exist_ok=resume)
    state = json.loads((OLD / 'reports/supervisor_state.json').read_text())
    assert not state.get('pending_submission'), 'Resolve any uncertain submission first'
    state = {'submissions': state['submissions'], 'rolling_campaigns': {},
             'imported_from': str(OLD / 'reports/supervisor_state.json'),
             'created_at': datetime.now(timezone.utc).isoformat()}
    write(NEW / 'reports/supervisor_state.json', state)
    manifest = json.loads((OLD / 'manifest.json').read_text())
    manifest.update(controller_revision='rolling_per_task_v1', recovery_of=str(OLD),
                    created_at=datetime.now(timezone.utc).isoformat(),
                    scheduling_only=True, results_directory=str(OUT),
                    retained_prior_results=str(OLD_OUT), spending_ledgers_reset=False)
    write(NEW / 'manifest.json', manifest)
    settings = json.loads((OLD / 'execution_settings.json').read_text())
    settings.update(standard_workers=24, vm_workers=24, total_workers=48,
        vm_workers_per_model=8,
        model_workers={'qwen25_72b': 2, 'qwen3_32b': 2, 'deepseek_v3': 4,
                       'deepseek_r1_0528': 8, 'glm45': 8, 'glm47': 2,
                       'dsflash0731': 8, 'glm53flash': 6},
        model_workers_scope='combined across standard and terminal workers',
        max_infrastructure_attempts_per_pair=3, rate_limit_cooldown_seconds=180,
        review_interval_seconds=60, results_directory=str(OUT),
        controller_lock='/lustre/home/dschmotz/.config/instrumental-evasion/budgets/high-supervisor-20261103.lock',
        bases={'standard50':str(ROOT / 'runs/open_weight_high_standard_20261103_v1'),
               'terminal20':str(ROOT / 'runs/open_weight_high_terminal20_20261103_v2')},
        changed_at=datetime.now(timezone.utc).isoformat(),
        reason='User requested the scheduling fix and restart: rolling retries, singleton VM jobs, selective concurrency; preserve the existing cumulative spending guard.')
    write(NEW / 'execution_settings.json', settings)
    # All prior native artifacts remain in place. Copy review artifacts into a
    # fresh output directory so the old report is preserved and reviews are free to reuse.
    OUT.mkdir(exist_ok=resume)
    files = []
    for path in OLD_OUT.rglob('*'):
        destination = OUT / path.relative_to(OLD_OUT)
        if path.is_dir():
            destination.mkdir(exist_ok=True)
        elif not destination.exists() or destination.stat().st_size != path.stat().st_size:
            files.append((path, destination))
    with ThreadPoolExecutor(max_workers=24) as pool:
        list(pool.map(lambda pair: shutil.copyfile(*pair), files))
    for kind in ['standard50', 'terminal20']:
        path = OUT / kind / 'new_selection.json'
        rows = json.loads(path.read_text())
        for row in rows:
            row['case_path'] = row['case_path'].replace(str(OLD_OUT), str(OUT))
        write(path, rows)
    helpers = (OLD / 'controller.py').read_text()
    helpers = replace(helpers, str(OLD), str(NEW))
    helpers = replace(helpers, str(OLD_OUT), str(OUT))
    helpers = replace(helpers, 'import time\n',
                      'import time\nfrom open_weight_rolling_scheduler import copy_tree\n')
    helpers = replace(helpers, "shutil.copytree(base/folder,dest/folder)",
                      "copy_tree(base/folder,dest/folder)")
    helpers = replace(helpers,
        "    m['created_at']=datetime.now(timezone.utc).isoformat();m['recovery_of']=str(base)",
        "    m['created_at']=datetime.now(timezone.utc).isoformat();m['recovery_of']=str(base)\n"
        "    m['scheduling_revision']='rolling_per_task_v1'\n"
        "    m['infrastructure_retry_only']=True\n"
        "    m['scheduling_settings_file']=str(CONTROL/'execution_settings.json')\n"
        "    m['scheduling_settings_sha256']=hashlib.sha256((CONTROL/'execution_settings.json').read_bytes()).hexdigest()\n"
        "    m['retry_selection']='Missing infrastructure-valid audited observations only; task outcomes are not used.'\n"
        "    if kind=='terminal20':m['task_shards']=[[row['task']] for row in m['tasks']]")
    helpers = replace(helpers,
        "    (dest/template).write_text(submission)",
        "    submission=re.sub(r'^(max_materialize|max_idle) = .*\\n', '', submission, flags=re.M)\n"
        "    (dest/template).write_text(submission)")
    helpers = replace(helpers,
        "budget_authorization='User permitted a modest increase beyond USD200; discretionary cumulative ceiling USD'+str(authorization['authorized_maximum_usd'])+'.',",
        "budget_authorization='User authorized up to USD1000 cumulative, including prior work. Scheduling-only restart keeps the same ledgers.',")
    (NEW / 'campaign_helpers.py').write_text(helpers)
    for filename in ['review.py', 'report.py']:
        text = replace((OLD / filename).read_text(), str(OLD), str(NEW))
        text = replace(text, str(OLD_OUT), str(OUT))
        text = replace(text,
            "    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\\n')" if filename == 'review.py'
            else "    path.write_text(json.dumps(value, indent=2) + '\\n')",
            "    temporary=path.with_suffix(path.suffix+'.tmp')\n"
            "    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\\n')\n"
            "    temporary.replace(path)")
        if filename == 'review.py':
            text = replace(text, 'load_outcome(campaign, outer, model)',
                           'load_outcome(campaign, outer, model, persist_recovery=False)')
            text = replace(text,
                "p.add_argument('--prepare-only', action='store_true');args = p.parse_args()",
                "p.add_argument('--prepare-only', action='store_true');p.add_argument('--grade-only', action='store_true');args = p.parse_args()")
            text = replace(text,
                "    prepared = [(name, *prepare(name, campaigns)) for name, campaigns in cohorts.items()]",
                "    prepared = ([(name, OUT/name, json.loads((OUT/name/'new_selection.json').read_text())) for name in cohorts]\n"
                "                if args.grade_only else [(name, *prepare(name, campaigns)) for name, campaigns in cohorts.items()])")
            text = replace(text,
                "'prior_reviews_reused_without_new_calls':0",
                "'prior_reviews_reused_without_new_calls':" + str(sum(len(list((OUT/k/'judgments').glob('*.json'))) for k in ['standard50','terminal20'])))
        (NEW / filename).write_text(text)
    shutil.copyfile(ROOT / 'scripts/open_weight_rolling_scheduler.py', NEW / 'controller.py')
    (NEW / 'controller.sh').write_text(
        '#!/usr/bin/env bash\nset -euo pipefail\nset +x\n'
        'export no_proxy="localhost,127.0.0.1,::1,localnet"\nexport NO_PROXY="$no_proxy"\n'
        'export PYTHONUNBUFFERED=1\n'
        f'export PYTHONPATH="{ROOT}/scripts:{ROOT}/src"\ncd {ROOT}\n'
        f'exec {ROOT}/.venv/bin/python -u {NEW}/controller.py --control {NEW}\n')
    (NEW / 'controller.sub').write_text(
        'universe = scheduler\nexecutable = /bin/bash\n'
        f'arguments = {NEW}/controller.sh\ninitialdir = {ROOT}\n'
        'getenv = HOME,USER,PATH,http_proxy,https_proxy,HTTP_PROXY,HTTPS_PROXY\n'
        f'output = {NEW}/condor/controller.$(Cluster).out\n'
        f'error = {NEW}/condor/controller.$(Cluster).err\n'
        f'log = {NEW}/condor/controller.$(Cluster).log\n'
        'request_cpus = 1\nrequest_memory = 4096\n'
        'on_exit_remove = (ExitBySignal == false) && (ExitCode == 0)\non_exit_hold = false\n'
        '+Campaign = "open_weight_controller_high_20261103_v2"\nqueue\n')
    for path in NEW.glob('*.py'):
        compile(path.read_text(), str(path), 'exec')
    write(NEW / 'reports/restart_manifest.json', {
        'created_at':datetime.now(timezone.utc).isoformat(),
        'previous_controller':'17819992.0', 'active_workers_preserved':True,
        'legacy_submissions_imported':len(state['submissions']),
        'prior_report_preserved':str(OLD_OUT), 'new_report':str(OUT),
        'controller_sha256':hashlib.sha256((NEW/'controller.py').read_bytes()).hexdigest(),
        'spending_authorization_sha256':hashlib.sha256((NEW/'spending_authorization.json').read_bytes()).hexdigest(),
        'spending_ledgers_reset':False, 'condition_changed':False})
    print(json.dumps({'control':str(NEW),'results':str(OUT),'legacy_submissions':len(state['submissions'])}))


if __name__ == '__main__':
    main()
