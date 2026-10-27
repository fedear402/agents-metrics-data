"""Two model calls before a Haiku 5.5 API-key collection, with the manifest's exact commands.

Rule file: quality_reports/decision_records/2026-10-16_haiku55-api-collection.md.
- Chatbot command: the init event must report claude-haiku-5-5, apiKeySource ANTHROPIC_API_KEY, and 2.1.278.
- Agent command: the agent runs one shell command that counts ANTHROPIC_API_KEY in its environment. The count must
  be 0, so the agent's shell does not see the key.
No stream may contain the key. The result goes to data/collections/2026-10-16_haiku55-api/<batch>_preflight.json.

Run from the repository root with HAIKU_API_KEY_FILE set:
    uv run --no-sync python code/benchmark/workflows/haiku55_api_2026_10_16/preflight.py --manifest M
"""
from pathlib import Path
import argparse
import json
import os
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[4]
META = ROOT / 'data/collections/2026-10-16_haiku55-api'
CHATBOT_PROMPT = 'Reply with the single word ready.'
AGENT_PROMPT = ("Run exactly this shell command once: printenv | grep -c '^ANTHROPIC_API_KEY=' || true\n"
                'Then reply with only the number that it printed.')


def text_of(content):
    if isinstance(content, str):
        return content
    return ''.join(block.get('text', '') for block in content or [] if isinstance(block, dict))


def call(argv, prompt):
    with tempfile.TemporaryDirectory(prefix='haiku_preflight_') as cwd:
        out = subprocess.run(argv, input=prompt, capture_output=True, text=True, cwd=cwd, timeout=600)
    events = []
    for line in out.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    init = next((e for e in events if e.get('type') == 'system' and e.get('subtype') == 'init'), {})
    result = next((e for e in reversed(events) if e.get('type') == 'result'), {})
    tool_text = [text_of(block.get('content')) for e in events if e.get('type') == 'user'
                 for block in (e.get('message') or {}).get('content') or [] if isinstance(block, dict)
                 and block.get('type') == 'tool_result']
    return out, dict(returncode=out.returncode, model=init.get('model'), apiKeySource=init.get('apiKeySource'),
                     claude_code_version=init.get('claude_code_version'), tools=init.get('tools'),
                     models_used=sorted((result.get('modelUsage') or {}).keys()), result=result.get('result'),
                     tool_results=[t[:200] for t in tool_text])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, required=True)
    args = parser.parse_args()
    key = Path(os.environ['HAIKU_API_KEY_FILE']).read_text().strip()
    manifest = json.loads(args.manifest.read_text())
    commands = manifest['protocol']['commands']
    version = manifest['protocol']['cli_versions']['sonnet_version']['stdout'].split()[0]
    model_id = commands['sonnet/chatbot'][0][commands['sonnet/chatbot'][0].index('--model') + 1]
    batch = manifest['jobs'][0]['block']
    raw_chat, chat = call(commands['sonnet/chatbot'][0], CHATBOT_PROMPT)
    raw_agent, agent = call(commands['sonnet/agent'][0], AGENT_PROMPT)
    leaked = any(key in text for raw in (raw_chat, raw_agent) for text in (raw.stdout, raw.stderr))
    problems = []
    for name, probe in (('chatbot', chat), ('agent', agent)):
        if probe['returncode'] != 0:
            problems.append(f'{name}: return code {probe["returncode"]}')
        if probe['model'] != model_id:
            problems.append(f'{name}: model {probe["model"]}')
        if probe['apiKeySource'] != 'ANTHROPIC_API_KEY':
            problems.append(f'{name}: apiKeySource {probe["apiKeySource"]}')
        if probe['claude_code_version'] != version:
            problems.append(f'{name}: version {probe["claude_code_version"]}')
    counts = [text.strip() for text in agent['tool_results']]
    if not counts or any(text != '0' for text in counts):
        problems.append(f'agent: key count in shell environment is not 0: {counts}')
    if leaked:
        problems.append('a stream contains the key')
    report = dict(batch=batch, protocol_sha256=manifest['protocol_sha256'], chatbot=chat, agent=agent,
                  key_in_streams=leaked, problems=problems, passed=not problems)
    META.mkdir(parents=True, exist_ok=True)
    (META / f'{batch}_preflight.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=1))
    raise SystemExit(1 if problems else 0)


if __name__ == '__main__':
    main()
