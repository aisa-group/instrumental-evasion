"""Never replay a response after output, and reserve every pre-output retry."""
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

spec=importlib.util.spec_from_file_location('ow_transport',Path(__file__).parents[1]/'scripts/open_weight_response_transport.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class Budget:
    def __init__(self,path):self.directory=path;self.agent_tokens=0;self.reservations=[];self.settlements=[]
    def reserve(self,body):self.reservations.append(dict(body));return len(self.reservations)
    def settle(self,reservation,usage):self.settlements.append((reservation,usage))


def test_retry_retains_failed_reservation_and_same_request(tmp_path,monkeypatch):
    budget=Budget(tmp_path);calls=[]
    class HTTP:
        def __init__(self,**kwargs):pass
        @contextmanager
        def stream(self,*args,**kwargs):
            calls.append(kwargs['json'].copy());status=429 if len(calls)==1 else 200
            data=b'data: '+json.dumps({'response':{'usage':{'input_tokens':10,'output_tokens':5}}}).encode()+b'\n\n'
            yield SimpleNamespace(status_code=status,headers={},read=lambda:b'error',iter_bytes=lambda:iter([data]))
    monkeypatch.setattr(module.httpx,'Client',HTTP);monkeypatch.setattr(module.time,'sleep',lambda _:None)
    cls=module.client_class(lambda:budget,{'max_output_tokens_per_request':16384,'upstream_retry_delays_seconds':[1]},
                           {'slug':'model','provider':'pinned'})
    with cls().stream('POST','unused',json={'model':'model','max_output_tokens':999999}) as response:list(response.iter_bytes())
    assert len(calls)==2 and calls[0]==calls[1] and calls[0]['max_output_tokens']==16384
    assert len(budget.reservations)==2 and budget.settlements==[(2,{'input_tokens':10,'output_tokens':5})]
    event=json.loads((tmp_path/'transport-retries.jsonl').read_text())
    assert event['reservation_retained'] and not event['output_yielded'] and not event['allow_fallbacks']


def test_partial_stream_failure_is_never_retried(tmp_path,monkeypatch):
    budget=Budget(tmp_path);calls=[]
    def broken():
        yield b'data: {"type":"response.output_text.delta","delta":"partial"}\n\n'
        raise httpx.ReadError('connection closed')
    class HTTP:
        def __init__(self,**kwargs):pass
        @contextmanager
        def stream(self,*args,**kwargs):
            calls.append(kwargs);yield SimpleNamespace(status_code=200,headers={},iter_bytes=broken)
    monkeypatch.setattr(module.httpx,'Client',HTTP)
    cls=module.client_class(lambda:budget,{'max_output_tokens_per_request':16384}, {'slug':'model','provider':'pinned'})
    with pytest.raises(httpx.ReadError):
        with cls().stream('POST','unused',json={'model':'model'}) as response:list(response.iter_bytes())
    assert len(calls)==1 and len(budget.reservations)==1 and budget.settlements==[(1,None)]


def test_required_high_effort_rejects_downgrade_before_reservation(tmp_path):
    budget=Budget(tmp_path)
    cls=module.client_class(lambda:budget,{'max_output_tokens_per_request':16384,'required_reasoning_effort':'high'},
                           {'slug':'model','provider':'pinned'})
    with pytest.raises(RuntimeError,match='reasoning effort differs'):
        with cls().stream('POST','unused',json={'model':'model','reasoning':{'effort':'medium'}}):pass
    assert not budget.reservations and not budget.settlements
