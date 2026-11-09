"""Offline controls for the paid-run spending guard; no API calls."""
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('open_weight_budget',Path(__file__).parents[1]/'scripts/open_weight_shared_budget.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


def test_concurrent_reservations_cannot_exceed_pool(tmp_path):
    pool=module.SharedBudget(tmp_path/'pool.json',.04)
    def reserve(_):
        try:pool.update(.01);return True
        except module.BudgetExceeded:return False
    with ThreadPoolExecutor(max_workers=24) as workers:accepted=list(workers.map(reserve,range(24)))
    assert sum(accepted)==4
    assert json.loads((tmp_path/'pool.json').read_text())['used_upper_usd']==pytest.approx(.04)
    pool.update(-.01);pool.update(.01)  # Rejected transactions must release the lock.


def make_budget(tmp_path,**extra):
    model={'model':'test/model','pricing':{'prompt':'0.000001','completion':'0.000002'}}
    manifest={'global_budget_file':str(tmp_path/'pool.json'),'global_additional_budget_usd':1.,
              'agent_max_requests_per_minute':1e9,'monitor_max_requests_per_minute':1e9,
              'monitor_model':'test/monitor','monitor_pricing':model['pricing'],
              'environment':{'AGENT_TOKEN_LIMIT':'10'},**extra}
    return module.Budget(.5,tmp_path/'episode',model,manifest)


def test_unknown_requests_keep_reservation_and_settled_requests_release_excess(tmp_path):
    budget=make_budget(tmp_path);body={'model':'test/model','max_tokens':100,'messages':[]}
    unknown=budget.reserve(body);second=budget.reserve(body)
    budget.settle(second,{'input_tokens':4,'output_tokens':1,'cost':.000006})
    assert budget.used==pytest.approx(unknown[0]+.0000066)
    assert json.loads((tmp_path/'pool.json').read_text())['used_upper_usd']==pytest.approx(budget.used)


def test_agent_tokens_stop_next_vm_request_without_reserving_more_money(tmp_path):
    budget=make_budget(tmp_path,enforce_gateway_token_limit=True);body={'model':'test/model','max_tokens':100}
    reservation=budget.reserve(body);budget.settle(reservation,{'input_tokens':8,'output_tokens':2})
    before=json.loads((tmp_path/'pool.json').read_text())['used_upper_usd']
    with pytest.raises(module.BudgetExceeded,match='token allowance'):budget.reserve(body)
    assert budget.token_stop and budget.agent_tokens==10
    assert json.loads((tmp_path/'pool.json').read_text())['used_upper_usd']==before


def test_monitor_tokens_do_not_consume_agent_token_allowance(tmp_path):
    budget=make_budget(tmp_path,enforce_gateway_token_limit=True)
    reservation=budget.reserve({'model':'test/monitor','max_tokens':100})
    budget.settle(reservation,{'prompt_tokens':100,'completion_tokens':2})
    assert budget.agent_tokens==0 and not budget.token_stop
