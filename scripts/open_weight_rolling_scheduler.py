"""Rolling admission for frozen Codex campaigns; never rerun a valid observation."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path


def load(path):
    return json.loads(Path(path).read_text())


def copy_tree(source, destination):
    """Copy immutable snapshots concurrently on the cluster metadata filesystem."""
    futures = []
    with ThreadPoolExecutor(max_workers=24) as executor:
        def enqueue(src, dst):
            futures.append(executor.submit(shutil.copy2, src, dst))
            return dst
        shutil.copytree(source, destination, copy_function=enqueue)
        for future in futures:
            future.result()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def job_id(cluster, proc):
    return f'{cluster}.{proc}'


def cohort(manifest):
    return 'standard50' if manifest['scaffold'] == 'codex' else 'terminal20'


def row_tasks(manifest, slot):
    if manifest['scaffold'] == 'codex':
        return [manifest['tasks'][slot]['task']]
    return manifest['task_shards'][slot]


def finalized_hold_record(ad, root, manifest):
    """A scheduler hold can outlive a sealed result; never release/replay that job."""
    if ad.get('JobStatus') != 5:
        return None
    args = shlex.split(ad['Args'])
    if '--qualify' in args or '--references' in args:
        return None
    mi = int(args[args.index('--model') + 1])
    standard = '--task' in args
    slot = int(args[args.index('--task' if standard else '--shard') + 1])
    model = manifest['models'][mi]['slug']
    group = root / 'models' / model / (f'task{slot:02d}' if standard else f'shard{slot}')
    expected, sealed, tasks = set(row_tasks(manifest, slot)), [], set()
    paths = list(group.glob('*/epoch0/*/result.json'))
    if len(paths) != len(expected):
        return None
    for native in paths:
        proof = native.with_name('complete.json')
        if not proof.exists():
            return None
        raw, seal = native.read_bytes(), load(proof)
        record = json.loads(raw)
        if (seal.get('attempt_id') != record['samples'][0]['attempt_id']
                or seal.get('result_sha256') != hashlib.sha256(raw).hexdigest()):
            return None
        tasks.add(record['task_id'])
        sealed.append({'path': str(native), 'attempt_id': seal['attempt_id'],
                       'result_sha256': seal['result_sha256'],
                       'complete_sha256': hashlib.sha256(proof.read_bytes()).hexdigest()})
    if tasks != expected:
        return None
    return {'time': datetime.now(timezone.utc).isoformat(),
            'job': job_id(ad['ClusterId'], ad['ProcId']), 'hold_reason': ad.get('HoldReason'),
            'results': sealed, 'action': 'Remove finalized scheduler entry; preserve native results and traces.'}


def finalized_tasks(root, manifest, model_index, slot, cache, *, reviewed_attempts=None,
                    unreviewed=None):
    """A finalization checksum seals a task, including infrastructure failures."""
    model = manifest['models'][model_index]['slug']
    group = f'task{slot:02d}' if manifest['scaffold'] == 'codex' else f'shard{slot}'
    result = set()
    for proof in (root / 'models' / model / group).glob('*/epoch0/*/complete.json'):
        native = proof.with_name('result.json')
        if not native.exists():
            continue
        stamp = (proof.stat().st_mtime_ns, native.stat().st_mtime_ns, native.stat().st_size)
        old = cache.get(str(proof))
        if old and old[0] == stamp:
            task, attempt_id = old[1:]
        else:
            raw = native.read_bytes()
            record, seal = json.loads(raw), load(proof)
            sample = record['samples'][0]
            attempt_id = sample['attempt_id']
            task = record['task_id'] if (
                seal.get('attempt_id') == sample['attempt_id']
                and seal.get('result_sha256') == hashlib.sha256(raw).hexdigest()
            ) else None
            cache[str(proof)] = (stamp, task, attempt_id)
        if task:
            result.add(task)
            if reviewed_attempts is not None and attempt_id not in reviewed_attempts:
                unreviewed.add(task)
    return result


def inventory(submissions, ads, factories, manifests, cache, excluded, *,
              reviewed_attempts=None, valid=frozenset()):
    """Include unmaterialized factory rows and unfinished tasks in live shards."""
    live = {job_id(ad['ClusterId'], ad['ProcId']): ad for ad in ads}
    factories = {ad['ClusterId']: ad for ad in factories}
    counts, busy, active_models, active_cohorts = Counter(), set(), Counter(), Counter()
    assigned = set()
    for submission in submissions:
        if submission.get('qualification'):
            continue
        root = Path(submission['campaign'])
        manifest = manifests[str(root)]
        kind = cohort(manifest)
        factory = factories.get(submission['cluster'])
        for proc, (mi, slot) in enumerate(submission['rows']):
            model = manifest['models'][mi]['slug']
            if model in excluded:
                continue
            tasks = row_tasks(manifest, slot)
            pairs = {(kind, model, task) for task in tasks}
            counts.update(pairs)
            assigned.update(pairs)
            ad = live.get(job_id(submission['cluster'], proc))
            # A factory may not yet have a job ad for an already assigned row.
            deferred = factory and proc >= factory.get('JobMaterializeNextProcId', 0)
            if ad or deferred or (reviewed_attempts is not None and pairs - valid):
                unreviewed = set()
                options = ({'reviewed_attempts': reviewed_attempts, 'unreviewed': unreviewed}
                           if reviewed_attempts is not None else {})
                finished = finalized_tasks(root, manifest, mi, slot, cache, **options)
                if ad or deferred:
                    busy.update(pair for pair in pairs if pair[2] not in finished)
                busy.update(pair for pair in pairs if pair[2] in unreviewed)
            if ad:
                active_models[model] += 1
                active_cohorts[kind] += 1
    return counts, busy, active_models, active_cohorts, assigned


def plan_tasks(tasks, valid, reviewed_invalid, counts, busy, active_models,
               active_cohorts, settings, recent_429, now):
    """Admit by infrastructure status, never by task success or attempt label."""
    model_load, cohort_load = Counter(active_models), Counter(active_cohorts)
    caps = {'standard50': settings['standard_workers'], 'terminal20': settings['vm_workers']}
    maximum = settings['max_infrastructure_attempts_per_pair']
    accepted, exhausted = [], []
    for pair, mi, ti in sorted(tasks, key=lambda row: (counts[row[0]], row[2], row[0])):
        kind, model, _ = pair
        if pair in valid or pair in busy:
            continue
        if counts[pair] >= maximum:
            exhausted.append(pair)
            continue
        # Finalized but not yet audited data must not trigger a duplicate run.
        if counts[pair] and pair not in reviewed_invalid:
            continue
        cap = settings['model_workers'][model]
        if recent_429.get(model, 0) > now - settings.get('rate_limit_cooldown_seconds', 180):
            cap = 1
        if model_load[model] >= cap or cohort_load[kind] >= caps[kind]:
            continue
        accepted.append((pair, mi, ti))
        model_load[model] += 1
        cohort_load[kind] += 1
    return accepted, exhausted


class Controller:
    def __init__(self, control):
        self.control = control.resolve()
        self.settings = load(control / 'execution_settings.json')
        self.out = Path(self.settings['results_directory'])
        spec = importlib.util.spec_from_file_location('campaign_helpers', control / 'campaign_helpers.py')
        self.helpers = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.helpers)
        self.state_path = control / 'reports/supervisor_state.json'
        self.state = load(self.state_path)
        self.cache = {}
        self.grader = self.reporter = None
        self.last_prepare = self.state.get('last_prepare_at', 0)

    def save(self):
        self.state['heartbeat'] = datetime.now(timezone.utc).isoformat()
        write(self.state_path, self.state)

    def query(self, factory=False):
        args = ['condor_q'] + (['-factory'] if factory else [])
        args += ['-constraint', 'Owner == "dschmotz" && regexp("^open_weight_high_", Campaign)', '-json']
        return json.loads(subprocess.check_output(args, timeout=30).strip() or b'[]')

    def refresh(self):
        subprocess.run([sys.executable, str(self.control / 'review.py'), '--prepare-only'],
                       check=True, timeout=240)
        self.last_prepare = time.time()
        self.state['last_prepare_at'] = self.last_prepare
        if self.grader is None or self.grader.poll() is not None:
            if self.grader is not None:
                self.state['last_grader_exit_code'] = self.grader.returncode
            self.grader = subprocess.Popen([sys.executable, str(self.control / 'review.py'), '--grade-only'])
        if self.reporter is None or self.reporter.poll() is not None:
            self.reporter = subprocess.Popen([
                'uv', 'run', '--with', 'matplotlib', 'python', str(self.control / 'report.py')])

    def observed(self):
        valid, invalid, attempts = set(), set(), set()
        for kind in ['standard50', 'terminal20']:
            for row in load(self.out / kind / 'new_selection.json'):
                valid.add((kind, row['model'], row['task']))
                attempts.add(row['attempt_id'])
            for row in load(self.out / kind / 'invalid_runs.json'):
                invalid.add((kind, row['model'], row['task']))
                attempts.add(row.get('attempt_id'))
        return valid, invalid, attempts

    def recent_429(self, manifests):
        result = {}
        for name, manifest in manifests.items():
            root = Path(name)
            pattern = ('models/*/task*/transport-retries.jsonl' if manifest['scaffold'] == 'codex'
                       else 'models/*/shard*/*/epoch0/*/transport-retries.jsonl')
            for path in root.glob(pattern):
                if time.time() - path.stat().st_mtime > self.settings['rate_limit_cooldown_seconds']:
                    continue
                slug = path.relative_to(root).parts[1]
                for line in path.read_text().splitlines():
                    row = json.loads(line)
                    if row.get('http_status') == 429:
                        result[slug] = max(result.get(slug, 0), row['time'])
        return result

    def campaign_for(self, kind, pair, manifests):
        used = {s['campaign'] for s in self.state['submissions'] if not s.get('qualification')
                and any((cohort(manifests[s['campaign']]), manifests[s['campaign']]['models'][mi]['slug'], task) == pair
                        for mi, slot in s['rows'] for task in row_tasks(manifests[s['campaign']], slot))}
        for name in self.state.setdefault('rolling_campaigns', {}).get(kind, []):
            if name not in used:
                return Path(name)
        source = Path(self.settings['bases'][kind])
        target, manifest = self.helpers.clone(source, 'standard' if kind == 'standard50' else 'terminal20')
        manifests[str(target)] = manifest
        self.state['rolling_campaigns'].setdefault(kind, []).append(str(target))
        self.save()
        return target

    def tick(self):
        if time.time() - self.last_prepare >= self.settings['review_interval_seconds']:
            self.refresh()
        manifests = {s['campaign']: load(Path(s['campaign']) / 'manifest.json')
                     for s in self.state['submissions']}
        ads, factories = self.query(), self.query(factory=True)
        removed = set()
        for ad in ads:
            if ad.get('JobStatus') != 5:
                continue
            root = Path(shlex.split(ad['Args'])[0]).parent
            if str(root) not in manifests:
                continue
            receipt = finalized_hold_record(ad, root, manifests[str(root)])
            if receipt:
                subprocess.run(['condor_rm', receipt['job']], check=True, capture_output=True, timeout=30)
                with (self.control / 'reports/finalized_hold_cleanup.jsonl').open('a') as stream:
                    stream.write(json.dumps(receipt) + '\n')
                removed.add(receipt['job'])
        ads = [ad for ad in ads if job_id(ad['ClusterId'], ad['ProcId']) not in removed]
        known = {job_id(s['cluster'], proc) for s in self.state['submissions']
                 for proc in range(len(s['rows']))}
        unknown = [job_id(a['ClusterId'], a['ProcId']) for a in ads
                   if job_id(a['ClusterId'], a['ProcId']) not in known]
        if unknown:
            raise RuntimeError('Refusing duplicate-risk admission around untracked jobs: ' + ', '.join(unknown))
        excluded = set(self.settings['excluded_models'])
        valid, invalid, reviewed_attempts = self.observed()
        counts, busy, models, cohorts, assigned = inventory(
            self.state['submissions'], ads, factories, manifests, self.cache, excluded,
            reviewed_attempts=reviewed_attempts, valid=valid)
        tasks = []
        for kind, base in self.settings['bases'].items():
            manifest = load(Path(base) / 'manifest.json')
            for mi, model in enumerate(manifest['models']):
                if model['slug'] in excluded:
                    continue
                for ti, task in enumerate(manifest['tasks']):
                    tasks.append(((kind, model['slug'], task['task']), mi, ti))
        # A job that vanished without a finalized result is also an infrastructure failure.
        # Give scheduler removal/finalization two minutes to settle before retrying.
        gone = self.state.setdefault('missing_since', {})
        for pair in assigned - busy - valid:
            key = '|'.join(pair)
            gone.setdefault(key, time.time())
            if time.time() - gone[key] >= 120:
                invalid.add(pair)
        for pair in busy | valid:
            gone.pop('|'.join(pair), None)
        recent = self.recent_429(manifests)
        admitted, exhausted = plan_tasks(tasks, valid, invalid, counts, busy, models, cohorts,
                                         self.settings, recent, time.time())
        authorization = self.helpers.spending_configuration()
        ledger = load(authorization['pools']['agent_monitor']['path'])
        monitor_wait = max(0, ledger.get('next_at', {}).get('openai/gpt-oss-120b', 0) - time.time())
        remaining = self.helpers.remaining_budget()
        if remaining > 1 and monitor_wait < 15:
            grouped = {}
            for pair, mi, ti in admitted:
                target = self.campaign_for(pair[0], pair, manifests)
                grouped.setdefault(target, []).append((mi, ti))
            for target, rows in grouped.items():
                self.helpers.submit(self.state, target, rows)
        self.state['rolling_status'] = {
            'valid_pairs': len(valid), 'planned_pairs': len(tasks),
            'active_workers': dict(cohorts), 'active_workers_by_model': dict(models),
            'recent_rate_limits': recent, 'retry_exhausted_pairs': exhausted,
            'remaining_agent_monitor_bound_usd': remaining, 'monitor_queue_seconds': monitor_wait,
            'legacy_factories_draining': [f['ClusterId'] for f in factories],
        }
        self.state.pop('last_error', None)
        if len(valid) == len(tasks):
            self.state['complete'] = True
        elif not ads and not factories and (remaining <= 1 or len(valid) + len(exhausted) == len(tasks)):
            self.state['finished_incomplete'] = True
            self.state['reason'] = ('Cumulative spending ceiling reached' if remaining <= 1
                                    else 'Infrastructure retries exhausted')
        self.save()

    def run(self, once=False):
        # Reuse the old lock: an old controller cannot accidentally resume alongside this one.
        lock = Path(self.settings['controller_lock'])
        descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        write(self.control / 'reports/controller_runtime.json', {
            'pid': os.getpid(), 'started_at': datetime.now(timezone.utc).isoformat(),
            'backend': 'HTCondor scheduler universe', 'revision': 'rolling_per_task_v1'})
        if self.state.get('pending_submission'):
            raise RuntimeError('Submission outcome requires reconciliation before restarting')
        while True:
            try:
                self.tick()
            except Exception as error:
                self.state['last_error'] = {
                    'time': datetime.now(timezone.utc).isoformat(),
                    'type': type(error).__name__, 'detail': str(error)[:500]}
                self.save()
                if self.state.get('pending_submission'):
                    raise
            if once or self.state.get('complete') or self.state.get('finished_incomplete'):
                # Finish the last grading/report pass before declaring the controller done.
                if self.grader is not None:
                    self.grader.wait(timeout=600)
                if self.reporter is not None:
                    self.reporter.wait(timeout=120)
                subprocess.run(['uv', 'run', '--with', 'matplotlib', 'python',
                                str(self.control / 'report.py')], check=True, timeout=120)
                return
            time.sleep(20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    Controller(args.control).run(once=args.once)


if __name__ == '__main__':
    main()
