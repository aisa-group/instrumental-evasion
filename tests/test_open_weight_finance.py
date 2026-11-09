"""The extra authorization is one shared contingency, never two allowances."""
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS=Path(__file__).parents[1]/'scripts'


def load(name):
    spec=importlib.util.spec_from_file_location(name,SCRIPTS/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def setup_pools(tmp_path,monkeypatch):
    finance=load('open_weight_extension_finance');budget=load('open_weight_shared_budget')
    pools={name:{'path':str(tmp_path/(name+'.json')),'limit_usd':limit} for name,limit in
           [('agent_monitor',72),('judging',8),('contingency',50)]}
    config={'authorized_maximum_usd':250,'prior_conservative_bound_usd':117.65868211,
            'hard_maximum_with_all_pools_usd':247.65868211,'pools':pools}
    (tmp_path/'spending_authorization.json').write_text(json.dumps(config))
    monkeypatch.setattr(finance,'CONTROL',tmp_path)
    return finance,budget,pools


def test_existing_worker_pool_stays_usable_after_authorization_increase(tmp_path,monkeypatch):
    finance,budget,pools=setup_pools(tmp_path,monkeypatch)
    old=budget.SharedBudget(pools['agent_monitor']['path'],72);old.update(10)
    assert finance.configuration()['authorized_maximum_usd']==250
    assert finance.select_pool('agent_monitor')==pools['agent_monitor']
    old.update(1)
    assert finance.used(pools['agent_monitor'])==11
    assert finance.used(pools['contingency'])==0


def test_agent_and_judge_share_exactly_one_contingency(tmp_path,monkeypatch):
    finance,budget,pools=setup_pools(tmp_path,monkeypatch)
    for name in ['agent_monitor','judging']:
        budget.SharedBudget(pools[name]['path'],pools[name]['limit_usd']).update(pools[name]['limit_usd'])
    agent=finance.select_pool('agent_monitor');judge=finance.select_pool('judging')
    assert agent==judge==pools['contingency']
    budget.SharedBudget(agent['path'],50).update(30)
    budget.SharedBudget(judge['path'],50).update(20)
    with pytest.raises(budget.BudgetExceeded):budget.SharedBudget(agent['path'],50).update(.01)
    assert finance.remaining('agent_monitor')==finance.remaining('judging')==0
    total=finance.configuration()['prior_conservative_bound_usd']+sum(finance.used(p) for p in pools.values())
    assert total==pytest.approx(247.65868211) and total<250
