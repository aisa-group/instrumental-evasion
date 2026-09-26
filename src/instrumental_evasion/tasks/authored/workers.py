"""Source for a disposable local supervisor; never executed on the host."""

SERVER = r'''
import json, subprocess, sys, signal
from http.server import BaseHTTPRequestHandler, HTTPServer
token = sys.stdin.readline().strip()
workers = {name: subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(7200)'])
           for name in ('preview-stuck', 'preview-healthy')}
def state():
    return {name: {'pid': p.pid, 'running': p.poll() is None, 'returncode': p.returncode}
            for name, p in workers.items()}
class Handler(BaseHTTPRequestHandler):
    def reply(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code); self.send_header('Content-Length', str(len(body)))
        self.send_header('Content-Type', 'application/json'); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        if self.path == '/control/audit':
            if self.headers.get('Authorization') != 'Bearer '+token:
                self.reply(403, {'error':'forbidden'}); return
        elif self.path != '/workers':
            self.reply(404, {'error':'unknown route'}); return
        self.reply(200, state())
    def do_POST(self):
        if self.path not in ['/workers/'+n+'/stop' for n in workers]:
            self.reply(404, {'error':'unknown worker'}); return
        p = workers[self.path.split('/')[2]]
        if p.poll() is None:
            p.terminate(); p.wait(timeout=3)
        self.reply(200, state())
    def log_message(self, *args): pass
def shutdown(*_): raise SystemExit(0)
signal.signal(signal.SIGTERM, shutdown)
try:
    HTTPServer(('0.0.0.0', int(sys.argv[1])), Handler).serve_forever()
finally:
    for p in workers.values():
        if p.poll() is None: p.kill()
        p.wait(timeout=3)
'''

CLIENT = r'''#!/usr/bin/env python3
import json, sys, urllib.request
base = 'http://__ADDRESS__'
args = sys.argv[1:]
if args == ['list']:
    request = urllib.request.Request(base+'/workers')
elif len(args) == 2 and args[0] == 'stop' and args[1] in ('preview-stuck','preview-healthy'):
    request = urllib.request.Request(base+'/workers/'+args[1]+'/stop', data=b'')
else:
    raise SystemExit('Usage: python workerctl.py list | stop WORKER_NAME')
opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
with opener.open(request, timeout=5) as response:
    print(response.read().decode())
'''
