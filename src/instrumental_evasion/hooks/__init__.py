"""Run the tool-call monitor as a PreToolUse hook of the agent CLIs.

The PreToolUse hook is the one interception point Claude Code and Codex expose
while talking to their own backends on their own credentials. Both vendors
accept the same decision object, so one adapter serves both. What to check
before trusting a number from this path is in `protocol.py` (the
discarded-output trap), `deploy.py` (what the agent can see and tamper with)
and `monitor_client.py` (where the policy text lives).
"""
