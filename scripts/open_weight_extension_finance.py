"""Budget selection for future runs; running workers keep their original pools."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTROL=ROOT/'runs/open_weight_extension_control_20261102_v1'
BUDGET_DIR=Path('/lustre/home/dschmotz/.config/instrumental-evasion/budgets')


def configuration():
    path=CONTROL/'spending_authorization.json'
    if path.exists():return json.loads(path.read_text())
    prior=json.loads((CONTROL/'manifest.json').read_text())
    return {'authorized_maximum_usd':200,'hard_maximum_with_all_pools_usd':prior['cumulative_bound_usd'],
        'pools':{
            'agent_monitor':{'path':prior['global_budget_file'],'limit_usd':72},
            'judging':{'path':str(BUDGET_DIR/'open_weight_extension_judging_20261102_v1.json'),'limit_usd':8}}}


def used(pool):
    path=Path(pool['path'])
    if not path.exists():return 0.
    state=json.loads(path.read_text())
    assert state['limit_usd']==pool['limit_usd'], 'Budget limit differs from its recorded authorization'
    return state['used_upper_usd']


def remaining(kind):
    pools=configuration()['pools']
    names=[kind]+(['contingency'] if 'contingency' in pools else [])
    return sum(max(0,pools[name]['limit_usd']-used(pools[name])) for name in names)


def select_pool(kind, *, minimum_remaining=1):
    pools=configuration()['pools'];primary=pools[kind]
    if primary['limit_usd']-used(primary)>minimum_remaining:return primary
    return pools.get('contingency',primary)
