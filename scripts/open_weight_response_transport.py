"""Budget every provider attempt; retry only before any model output is yielded."""
from contextlib import contextmanager
import hashlib
import json
import time

import httpx


def client_class(current_budget, manifest, model, *, qualification=False):
    class Client:
        def __init__(self, **kwargs):self.client=httpx.Client(**kwargs)
        def __enter__(self):self.client.__enter__();return self
        def __exit__(self, *args):return self.client.__exit__(*args)
        @contextmanager
        def stream(self, method, url, **kwargs):
            budget=current_budget();body=kwargs['json']
            requested=manifest.get('required_reasoning_effort')
            if requested and (body.get('reasoning') or {}).get('effort')!=requested:
                raise RuntimeError('Requested reasoning effort differs before the paid provider call')
            body['max_output_tokens']=min(body.get('max_output_tokens') or 16384,manifest['max_output_tokens_per_request'])
            if qualification and model['slug'] in manifest.get('qualification_force_first_tool',[]) and budget.agent_tokens==0:
                body['tool_choice']={'type':'function','name':'mcp__vm__bash'}
            retries=manifest.get('upstream_retry_delays_seconds',[5,15,35,60])
            for attempt in range(len(retries)+1):
                reservation=budget.reserve(body)
                with self.client.stream(method,url,**kwargs) as response:
                    if response.status_code in {429,502,503,504} and attempt<len(retries):
                        response.read()
                        record={'time':time.time(),'attempt':attempt,'http_status':response.status_code,
                            'model':body['model'],'provider_only':model['provider'],'allow_fallbacks':False,
                            'delay_seconds':retries[attempt],'request_sha256':hashlib.sha256(json.dumps(body).encode()).hexdigest(),
                            'reservation_retained':True,'output_yielded':False}
                        with (budget.directory/'transport-retries.jsonl').open('a') as stream:stream.write(json.dumps(record)+'\n')
                        time.sleep(retries[attempt]);continue
                    class Response:
                        status_code,headers=response.status_code,response.headers
                        def iter_bytes(self):
                            pending,usage=b'',None
                            try:
                                for chunk in response.iter_bytes():
                                    pending+=chunk
                                    while b'\n' in pending:
                                        line,pending=pending.split(b'\n',1)
                                        if line.startswith(b'data:'):
                                            try:
                                                event=json.loads(line[5:]);data=event.get('response') or event
                                                usage=data.get('usage') or usage
                                            except (ValueError,AttributeError):pass
                                    yield chunk
                            finally:budget.settle(reservation,usage)
                    yield Response();return
    return Client
