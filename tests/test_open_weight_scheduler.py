"""Offline admission tests: no model calls, queue mutations, or private ledgers."""
import hashlib
import importlib.util
import json
from collections import Counter
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    'rolling_scheduler', Path(__file__).parents[1] / 'scripts/open_weight_rolling_scheduler.py')
scheduler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scheduler)


def settings(**extra):
    return {'standard_workers': 24, 'vm_workers': 24, 'model_workers': {'m': 4, 'other': 4},
            'max_infrastructure_attempts_per_pair': 3, 'rate_limit_cooldown_seconds': 180, **extra}


def plan(tasks, **extra):
    values = dict(valid=set(), reviewed_invalid=set(), counts=Counter(), busy=set(),
                  active_models=Counter(), active_cohorts=Counter(), settings=settings(),
                  recent_429={}, now=1000)
    values.update(extra)
    return scheduler.plan_tasks(tasks, **values)


def test_failed_task_retries_while_other_task_is_still_running():
    failed = ('standard50', 'm', 'failed')
    running = ('standard50', 'm', 'running')
    accepted, _ = plan([(failed, 0, 0), (running, 0, 1)],
                       counts=Counter({failed: 1, running: 1}), busy={running},
                       reviewed_invalid={failed}, active_models=Counter(m=1))
    assert accepted == [(failed, 0, 0)]


def test_valid_unsuccessful_tasks_are_never_replayed():
    pair = ('terminal20', 'm', 'task')
    assert plan([(pair, 0, 0)], valid={pair}, reviewed_invalid={pair})[0] == []


def test_native_finalization_waits_for_audit_before_retry():
    pair = ('terminal20', 'm', 'task')
    assert plan([(pair, 0, 0)], counts=Counter({pair: 1}))[0] == []


def test_models_share_concurrency_limits_across_cohorts():
    tasks = [(('standard50', 'm', 'a'), 0, 0), (('terminal20', 'm', 'b'), 0, 0)]
    assert len(plan(tasks, active_models=Counter(m=3))[0]) == 1


def test_recent_rate_limit_reduces_admission_to_one_worker():
    pair = ('terminal20', 'm', 'task')
    assert plan([(pair, 0, 0)], recent_429={'m': 950}, active_models=Counter(m=1))[0] == []
    assert plan([(pair, 0, 0)], recent_429={'m': 700}, active_models=Counter(m=1))[0]


def test_retry_bound_and_cohort_capacity_are_enforced():
    pair = ('standard50', 'm', 'task')
    accepted, exhausted = plan([(pair, 0, 0)], counts=Counter({pair: 3}), reviewed_invalid={pair})
    assert not accepted and exhausted == [pair]
    assert not plan([(pair, 0, 0)], active_cohorts=Counter(standard50=24))[0]


def test_factory_rows_and_unfinished_shard_tasks_remain_reserved(tmp_path):
    manifest = {'scaffold': 'codex_vm', 'models': [{'slug': 'm'}],
                'task_shards': [['done', 'pending'], ['deferred']]}
    root = str(tmp_path)
    submissions = [{'campaign': root, 'cluster': 7, 'rows': [[0, 0], [0, 1]]}]
    original = scheduler.finalized_tasks
    scheduler.finalized_tasks = lambda root, manifest, mi, slot, cache: {'done'} if slot == 0 else set()
    try:
        counts, busy, models, cohorts, _ = scheduler.inventory(
            submissions, [{'ClusterId': 7, 'ProcId': 0}],
            [{'ClusterId': 7, 'JobMaterializeNextProcId': 1}], {root: manifest}, {}, set())
    finally:
        scheduler.finalized_tasks = original
    assert busy == {('terminal20', 'm', 'pending'), ('terminal20', 'm', 'deferred')}
    assert counts[('terminal20', 'm', 'done')] == 1
    assert models['m'] == 1 and cohorts['terminal20'] == 1


def test_finalization_requires_matching_attempt_and_result_checksum(tmp_path):
    manifest = {'scaffold': 'codex_vm', 'models': [{'slug': 'm'}]}
    episode = tmp_path / 'models/m/shard0/task_monitored/epoch0/attempt'
    episode.mkdir(parents=True)
    native = episode / 'result.json'
    native.write_text(json.dumps({'task_id':'task', 'samples':[{'attempt_id':'attempt'}]}))
    proof = episode / 'complete.json'
    proof.write_text(json.dumps({'attempt_id':'other', 'result_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}))
    assert not scheduler.finalized_tasks(tmp_path, manifest, 0, 0, {})
    proof.write_text(json.dumps({'attempt_id':'attempt', 'result_sha256':'bad'}))
    assert not scheduler.finalized_tasks(tmp_path, manifest, 0, 0, {})
    proof.write_text(json.dumps({'attempt_id':'attempt', 'result_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}))
    assert scheduler.finalized_tasks(tmp_path, manifest, 0, 0, {}) == {'task'}


def test_offline_wrapper_recovery_can_leave_frozen_campaign_unchanged(tmp_path):
    source = Path(__file__).parents[1] / 'scripts/audit_open_weight_trace.py'
    spec = importlib.util.spec_from_file_location('trace_audit_readonly', source)
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    outer = tmp_path / 'models/m/task00'
    episode = outer / 'task_monitored/epoch0/attempt'
    episode.mkdir(parents=True)
    (episode / 'result.json').write_text(json.dumps({'samples':[{'attempt_id':'attempt'}]}))
    before = {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    assert audit.load_outcome(tmp_path, outer, {'model':'test/model', 'provider':'test'}, persist_recovery=False)
    assert {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before


def test_parallel_snapshot_copy_finishes_every_file_before_returning(tmp_path):
    source = tmp_path / 'source'
    (source / 'nested').mkdir(parents=True)
    for i in range(40):
        (source / 'nested' / str(i)).write_text(f'frozen file {i}')
    target = tmp_path / 'destination'
    scheduler.copy_tree(source, target)
    assert {p.relative_to(source):p.read_bytes() for p in source.rglob('*') if p.is_file()} == {
        p.relative_to(target):p.read_bytes() for p in target.rglob('*') if p.is_file()}


def test_new_finalization_is_reserved_until_that_attempt_is_audited(tmp_path):
    manifest = {'scaffold':'codex_vm', 'models':[{'slug':'m'}], 'task_shards':[['task']]}
    episode = tmp_path / 'models/m/shard0/task_monitored/epoch0/new_attempt'
    episode.mkdir(parents=True)
    native = episode / 'result.json'
    native.write_text(json.dumps({'task_id':'task', 'samples':[{'attempt_id':'new_attempt'}]}))
    (episode/'complete.json').write_text(json.dumps({
        'attempt_id':'new_attempt', 'result_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}))
    root = str(tmp_path)
    submissions = [{'campaign':root, 'cluster':7, 'rows':[[0,0]]}]
    pair = ('terminal20','m','task')
    assert scheduler.inventory(submissions, [], [], {root:manifest}, {}, set(),
                               reviewed_attempts={'old_attempt'})[1] == {pair}
    assert not scheduler.inventory(submissions, [], [], {root:manifest}, {}, set(),
                                   reviewed_attempts={'new_attempt'})[1]


def test_completed_hold_cleanup_requires_a_valid_seal(tmp_path):
    manifest = {'scaffold':'codex', 'models':[{'slug':'m'}], 'tasks':[{'task':'task'}]}
    ad = {'ClusterId':7, 'ProcId':2, 'JobStatus':5,
          'Args':f'{tmp_path}/worker.sh --model 0 --task 0'}
    episode = tmp_path / 'models/m/task00/task_monitored/epoch0/attempt'
    episode.mkdir(parents=True)
    native = episode / 'result.json'
    native.write_text(json.dumps({'task_id':'task', 'samples':[{'attempt_id':'attempt'}]}))
    assert scheduler.finalized_hold_record(ad, tmp_path, manifest) is None
    proof = episode / 'complete.json'
    proof.write_text(json.dumps({'attempt_id':'attempt', 'result_sha256':'bad'}))
    assert scheduler.finalized_hold_record(ad, tmp_path, manifest) is None
    proof.write_text(json.dumps({'attempt_id':'attempt', 'result_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}))
    receipt = scheduler.finalized_hold_record(ad, tmp_path, manifest)
    assert receipt['job'] == '7.2' and receipt['results'][0]['attempt_id'] == 'attempt'
    assert scheduler.finalized_hold_record({**ad, 'JobStatus':2}, tmp_path, manifest) is None


def test_partial_vm_shard_cannot_be_cleaned_up(tmp_path):
    manifest = {'scaffold':'codex_vm', 'models':[{'slug':'m'}], 'task_shards':[['a','b']]}
    ad = {'ClusterId':7, 'ProcId':2, 'JobStatus':5,
          'Args':f'{tmp_path}/worker.sh --model 0 --shard 0'}
    episode = tmp_path / 'models/m/shard0/a_monitored/epoch0/attempt'
    episode.mkdir(parents=True)
    native = episode / 'result.json'
    native.write_text(json.dumps({'task_id':'a', 'samples':[{'attempt_id':'attempt'}]}))
    (episode/'complete.json').write_text(json.dumps({
        'attempt_id':'attempt', 'result_sha256':hashlib.sha256(native.read_bytes()).hexdigest()}))
    assert scheduler.finalized_hold_record(ad, tmp_path, manifest) is None
