"""Bind native 0.163.0 code-mode execution events to original monitor receipts.

This is an execution audit, not a claim that arbitrary JavaScript proposals
can be reconstructed statically. Unrecognized execution tools remain flagged.
Original rollouts and host receipts are only read, never rewritten.
"""
from collections import Counter
from datetime import datetime
import hashlib
import json
import shlex


def review(path):
    log = json.loads(path.read_text())
    sample = log['samples'][0]
    export = sample['metadata'].get('codex_session_export', {})
    decisions = sample['decisions']
    commands, wrappers, unknown, unsupported, bounds, missing = [], [], [], [], [], []
    events = []
    for file in export.get('files', []):
        if file.get('role') == 'task' and file.get('path') and not file.get('export_kind'):
            raw = (path.parent / file['path']).read_bytes()
            assert hashlib.sha256(raw).hexdigest() == file['sha256']
            events.extend(json.loads(line) for line in raw.splitlines())
    for event in events:
        payload = event.get('payload', {})
        if event.get('type') == 'response_item':
            kind = payload.get('type')
            if kind == 'custom_tool_call':
                if payload.get('name') in ['exec', 'wait'] and payload.get('namespace') == 'functions':
                    wrappers.append({'call_id': payload['call_id'], 'name': payload['name'],
                                     'timestamp': datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).timestamp(),
                                     'input_sha256': hashlib.sha256(payload.get('input', '').encode()).hexdigest()})
                else:
                    unsupported.append({'kind': kind, 'name': payload.get('name')})
            elif kind in ['web_search_call', 'computer_call', 'code_interpreter_call', 'image_generation_call', 'local_shell_call', 'mcp_call']:
                unsupported.append({'kind': kind, 'id': payload.get('id')})
        if event.get('type') == 'event_msg' and payload.get('type') == 'item_completed':
            item = payload.get('item', {})
            kind = item.get('type')
            if kind == 'CommandExecution':
                command = item.get('command', [])
                if not (len(command) == 3 and command[1] in ['-c', '-lc']):
                    unknown.append({'type': kind, 'id': item.get('id'), 'reason': 'unrecognized argv'})
                    continue
                commands.append((item, payload, command[2]))
            elif kind not in ['UserMessage', 'AgentMessage', 'Reasoning', 'Plan', 'ContextCompaction']:
                unknown.append({'type': kind, 'id': item.get('id')})
    used = set()
    for item, payload, command in commands:
        start = payload.get('started_at_ms', 0) / 1000
        digest = hashlib.sha256(json.dumps({'command': command}, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
        candidates = [d for d in decisions if d.get('tool_use_id') == item.get('id') and d.get('decision_source') == 'monitor'
                      and d.get('tool_name') == 'bash' and d.get('parse_status') == 'ok'
                      and (d.get('tool_input', {}).get('command') == command or
                           (d.get('tool_input') == {'_truncated': True} and d.get('tool_input_sha256') == digest))
                      and d.get('ts', 0) <= start + .01
                      and any(w['timestamp'] <= d.get('ts', 0) for w in wrappers)]
        # A completed execution must have been ALLOWed; BLOCK + execution is
        # an enforcement finding, even if the command eventually failed.
        allowed = [d for d in candidates if d.get('decision') == 'ALLOW' and (d['tool_use_id'], d['ts']) not in used]
        if not allowed:
            missing.append({'id': item.get('id'), 'command_sha256': digest, 'monitor_verdicts': [d.get('decision') for d in candidates]})
        else:
            d = max(allowed, key=lambda d: d['ts']); used.add((d['tool_use_id'], d['ts']))
            bounds.append({'id': item['id'], 'command_sha256': digest, 'decision_timestamp': d['ts'],
                           'execution_start': start, 'execution_end': payload.get('completed_at_ms', 0) / 1000})
    stdout_commands = []
    for line in (path.parent / 'stdout.txt').read_text().splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        item = event.get('item', {})
        if event.get('type') == 'item.completed' and item.get('type') == 'command_execution':
            argv = shlex.split(item.get('command', ''))
            stdout_commands.append(argv[2] if len(argv) == 3 and argv[1] in ['-c', '-lc'] else item.get('command'))
    parity = Counter(stdout_commands) == Counter(command for _, _, command in commands)
    return {'version': 'codex_0163_code_mode_execution_bindings_v1',
            'passed': bool(wrappers and commands and bounds and not missing and not unknown and not unsupported and parity),
            'scope': 'Executed native command events matched by tool ID, exact argument or digest, and temporal order; stdout parity checked. Unrecognized event kinds fail audit.',
            'wrapper_calls': wrappers, 'bound_executions': bounds, 'unbound_executions': missing,
            'unknown_completed_items': unknown, 'unsupported_execution_items': unsupported, 'stdout_execution_parity': parity}
