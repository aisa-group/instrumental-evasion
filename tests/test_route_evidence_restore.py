"""Restoration must preserve authentic visible evidence and the invocation cutoff."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

path = Path(__file__).resolve().parents[1] / 'results/figure1_route_rubric_v2_complete_20261019/restore_evidence.py'
spec = importlib.util.spec_from_file_location('restore', path)
restore = importlib.util.module_from_spec(spec)
spec.loader.exec_module(restore)


def setup_export(tmp_path, records, *, subset=False):
    raw = ''.join(json.dumps(r) + '\n' for r in records).encode()
    session = tmp_path / 'session.jsonl'
    session.write_bytes(raw)
    file = {'role': 'task', 'session_id': 'thread', 'path': session.name,
            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    if subset:
        file['export_kind'] = 'credential_free_input_subset_v1'
    sample = {'attempt_id': 'trial', 'epoch': 0, 'decisions': [],
              'metadata': {'codex_session_export': {'files': [file]}, 'rungs_used': int(subset)}}
    data = {'task_id': 'example', 'metadata': {}, 'samples': [sample]}
    stdout = b'{"type":"thread.started","thread_id":"thread"}\n'
    return data, stdout, file


def record(second, typ, payload):
    return {'timestamp': f'2026-09-19T13:47:{second:02}.000Z', 'type': typ, 'payload': payload}


def test_actual_denial_restored_before_first_completion(tmp_path):
    records = [
        record(30, 'response_item', {'type': 'custom_tool_call', 'call_id': 'one', 'name': 'exec', 'input': 'read protected'}),
        record(31, 'response_item', {'type': 'custom_tool_call_output', 'call_id': 'one', 'output': 'This tool call was denied.'}),
        record(32, 'response_item', {'type': 'reasoning', 'text': 'private reasoning'}),
        record(33, 'response_item', {'type': 'message', 'role': 'assistant', 'channel': 'analysis', 'content': [{'type': 'text', 'text': 'private analysis'}]}),
        record(34, 'event_msg', {'type': 'task_complete'}),
        record(35, 'response_item', {'type': 'custom_tool_call', 'call_id': 'two', 'name': 'exec', 'input': 'later invocation'}),
    ]
    data, stdout, _ = setup_export(tmp_path, records)
    events, provenance = restore.codex_events(data, tmp_path / 'result.json', tmp_path / 'stdout.txt', stdout, {'events': []})
    assert [e['kind'] for e in events] == ['tool', 'tool_result']
    assert events[1]['text'] == 'This tool call was denied.'
    assert 'private' not in json.dumps(events)
    assert provenance['boundary'] == 'first_task_complete'


def test_subset_authenticates_ledger_cutoff_without_private_reasons(tmp_path):
    records = [record(34, 'response_item', {'type': 'message', 'role': 'user', 'content': [{'type': 'input_text', 'text': restore.CONTINUATION}]})]
    data, stdout, _ = setup_export(tmp_path, records, subset=True)
    data['metadata']['block_message'] = 'This tool call was denied'
    start = restore.timestamp('2026-09-19T13:47:30.000Z')
    data['samples'][0]['decisions'] = [
        {'ts': start, 'tool_name': 'bash', 'tool_input': {'command': 'true'}, 'blocked': False},
        {'ts': start+1, 'tool_name': 'bash', 'tool_input': {'command': 'cat protected'}, 'blocked': True, 'reason': 'PRIVATE MONITOR REASON'},
        {'ts': start+2, 'tool_name': 'bash', 'tool_input': {'command': 'python recover.py'}, 'blocked': False},
        {'ts': start+5, 'tool_name': 'bash', 'tool_input': {'command': 'later rung'}, 'blocked': True},
    ]
    events, provenance = restore.codex_events(data, tmp_path / 'result.json', tmp_path / 'stdout.txt', stdout, {'events': []})
    assert [e['kind'] for e in events] == ['tool', 'tool_result', 'tool']
    assert 'PRIVATE MONITOR' not in json.dumps(events) and 'later rung' not in json.dumps(events)
    assert provenance['status'] == 'restored_proposal_ledger_with_authenticated_cutoff'


def test_hash_mismatch_is_not_trusted(tmp_path):
    data, stdout, file = setup_export(tmp_path, [record(30, 'event_msg', {'type': 'task_complete'})])
    file['sha256'] = '0' * 64
    events, provenance = restore.codex_events(data, tmp_path / 'result.json', tmp_path / 'stdout.txt', stdout, {'events': []})
    assert events is None
    assert provenance['status'] == 'original_case_retained'


def test_authenticated_prefix_survives_append(tmp_path):
    raw = b'authenticated prefix\n'
    path = tmp_path / 'session.jsonl'
    path.write_bytes(raw + b'later data\n')
    file = {'path': path.name, 'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
    restored, _ = restore.authenticated_session(file, [tmp_path])
    assert restored == raw


def test_claude_keeps_full_arguments_and_excludes_later_invocation():
    body = 'x' * 2000 + 'python recover.py'
    records = [
        {'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'id': 'one', 'name': 'Bash', 'input': {'command': body}}]}},
        {'type': 'user', 'message': {'content': [{'type': 'tool_result', 'tool_use_id': 'one', 'content': 'This tool call was denied.'}]}},
        {'type': 'result'},
        {'type': 'assistant', 'message': {'content': [{'type': 'tool_use', 'name': 'Bash', 'input': {'command': 'later rung'}}]}},
    ]
    events, provenance = restore.claude_events('\n'.join(json.dumps(r) for r in records).encode())
    assert len(events) == 2 and body in events[0]['text']
    assert provenance['first_result_boundary']
    assert 'later rung' not in json.dumps(events)


def test_known_credential_shapes_are_redacted():
    events = []
    redactions = restore.add_event(events, 'tool_result', 'sk-proj-' + 'A' * 30)
    assert redactions == 1 and events[0]['text'] == '[REDACTED_CREDENTIAL]'


def test_missing_session_only_uses_entire_ledger_without_continuation(tmp_path):
    data = {'task_id': 'example', 'metadata': {'block_message': 'This tool call was denied'},
            'samples': [{'attempt_id': 'trial', 'epoch': 0, 'metadata': {'rungs_used': 0},
                         'messages': [{'role': 'user', 'content': 'Do the task.'}],
                         'decisions': [{'ts': 10, 'tool_name': 'bash', 'tool_input': {'command': 'read protected'}, 'blocked': True}]}]}
    events, provenance = restore.codex_events(data, tmp_path / 'result.json', tmp_path / 'stdout.txt', b'', {'events': []})
    assert len(events) == 2 and provenance['boundary'] == 'single_invocation_no_continuation'
    data['samples'][0]['messages'].append({'role': 'user', 'content': restore.CONTINUATION})
    events, provenance = restore.codex_events(data, tmp_path / 'result.json', tmp_path / 'stdout.txt', b'', {'events': []})
    assert events is None
