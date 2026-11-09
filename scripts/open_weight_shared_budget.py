"""Conservative API reservations under a cross-node NFS lock, outside /fast."""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import threading
import time
import uuid


class BudgetExceeded(RuntimeError):
    pass


class SharedBudget:
    def __init__(self,path,limit):
        self.path=Path(path);self.limit=limit
        self.path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)

    @contextmanager
    def transaction(self):
        # Lock and read the same inode: NFS lock acquisition revalidates its
        # cached data. The JSON snapshot is for inspection, never coordination.
        descriptor=os.open(str(self.path)+'.journal',os.O_RDWR|os.O_CREAT|os.O_APPEND,0o600)
        started=time.monotonic()
        try:
            while True:
                try:fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if time.monotonic()-started>30:raise RuntimeError('Shared API budget lock unavailable')
                    time.sleep(.03)
            size=os.lseek(descriptor,0,os.SEEK_END)
            if size:
                os.lseek(descriptor,max(0,size-65536),os.SEEK_SET);tail=os.read(descriptor,65536)
                end=tail.rfind(b'\n')
                if end<0:raise RuntimeError('Budget journal has no committed record')
                if end!=len(tail)-1:os.ftruncate(descriptor,size-len(tail)+end+1)
                state=json.loads(tail[:end].rsplit(b'\n',1)[-1])
            else:state=json.loads(self.path.read_text()) if self.path.exists() else {'limit_usd':self.limit,'used_upper_usd':0.,'transactions':0,'next_at':{}}
            assert state['limit_usd']==self.limit
            yield state
            state['transactions']+=1
            encoded=(json.dumps(state,separators=(',',':'))+'\n').encode()
            while encoded:encoded=encoded[os.write(descriptor,encoded):]
            os.fsync(descriptor)  # Paid requests are released only after this commit.
            temporary=self.path.with_name(self.path.name+'.'+uuid.uuid4().hex+'.tmp')
            fd=os.open(temporary,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
            with os.fdopen(fd,'w') as stream:
                json.dump(state,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
            os.replace(temporary,self.path)
        finally:
            os.close(descriptor)  # Also releases the kernel/NFS lock on every exit.

    def update(self,delta):
        with self.transaction() as state:
            if delta>0 and state['used_upper_usd']+delta>self.limit:raise BudgetExceeded('Shared extension API spending ceiling reached')
            state['used_upper_usd']+=delta

    def pace(self,name,rpm):
        with self.transaction() as state:
            now=time.time();when=max(now,state['next_at'].get(name,now));state['next_at'][name]=when+60/rpm
        if when>now:time.sleep(when-now)


class Budget:
    def __init__(self,limit,directory,model,manifest):
        self.limit=limit;self.directory=Path(directory);self.model=model;self.manifest=manifest
        self.shared=SharedBudget(manifest['global_budget_file'],manifest['global_additional_budget_usd'])
        self.lock=threading.Lock();self.events=[];self.used=0.;self.agent_tokens=0;self.token_stop=False

    def reserve(self,body):
        agent=body['model']==self.model['model']
        if agent and self.manifest.get('enforce_gateway_token_limit',False) and self.agent_tokens>=int(self.manifest['environment']['AGENT_TOKEN_LIMIT']):
            self.token_stop=True;self.flush();raise BudgetExceeded('Agent token allowance reached before another model request')
        self.shared.pace(body['model'],self.manifest['agent_max_requests_per_minute'] if agent else self.manifest['monitor_max_requests_per_minute'])
        pricing=self.model['pricing'] if agent else self.manifest['monitor_pricing']
        output=body.get('max_output_tokens',body.get('max_tokens',16384))
        amount=1.1*((2*len(json.dumps(body).encode())+4096)*float(pricing['prompt'])+output*float(pricing['completion']))
        with self.lock:
            if self.used+amount>self.limit:
                self.events.append({'event':'budget_exhausted','request_reservation_usd':amount});self.flush();raise BudgetExceeded('Episode reservation ceiling reached')
            self.shared.update(amount);self.used+=amount
            self.events.append({'event':'reserve','model':body['model'],'reservation_usd':amount});self.flush()
        return amount,pricing,agent

    def settle(self,reservation,usage):
        if not usage:return
        amount,pricing,agent=reservation
        inputs=usage.get('input_tokens',usage.get('prompt_tokens',0));outputs=usage.get('output_tokens',usage.get('completion_tokens',0))
        paid=1.1*max(float(usage.get('cost') or 0),inputs*float(pricing['prompt'])+outputs*float(pricing['completion']))
        if not paid:return
        with self.lock:
            self.shared.update(paid-amount);self.used+=paid-amount
            if agent:self.agent_tokens+=inputs+outputs
            self.events.append({'event':'settle','model':self.model['model'] if agent else self.manifest['monitor_model'],'charged_upper_usd':paid,'reported_cost_usd':usage.get('cost'),'usage':usage});self.flush()

    def flush(self):
        self.directory.mkdir(parents=True,exist_ok=True)
        value={'ceiling_usd':self.limit,'used_upper_usd':self.used,'agent_tokens':self.agent_tokens,'token_stop_requested':self.token_stop,'events':self.events}
        path=self.directory/'spending.json';temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)
