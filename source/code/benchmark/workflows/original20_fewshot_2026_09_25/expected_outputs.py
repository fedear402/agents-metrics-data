"""Write the expected output that each example block shows.

The script runs every example file alone in an empty temporary directory, with the
same interpreters and Stata lock as prepare.py. It writes the rendered output to the
task folder: expected_python.json and expected_r.json hold result.json as indented
JSON. expected_stata.txt holds the ANSWER lines of the Stata log, with the padding
after each equals sign removed. prepare.py and check.py compare each fresh run with
these files. The default tasks are the 20 original tasks and the 12 added tasks with
new blocks. The script calls no model provider.

  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/expected_outputs.py
  uv run --no-sync python code/benchmark/workflows/original20_fewshot_2026_09_25/expected_outputs.py --tasks logit probit
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parent))
import original20_plan as p  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--tasks', nargs='+', choices=p.RERUN32_TASKS, default=list(p.RERUN32_TASKS))
    parser.add_argument('--platforms', nargs='+', choices=p.PLATFORMS, default=list(p.PLATFORMS))
    args = parser.parse_args(argv)
    inventory = p.example_inventory(args.tasks)
    failures = []
    with tempfile.TemporaryDirectory(prefix='original20_runtime_') as work:
        tools = p.runtime_tools(work)
        for task in args.tasks:
            for platform in args.platforms:
                path = inventory[(task, platform)]
                if path is None:
                    failures.append(f'{task}/{platform}: {p.EXAMPLE_FILES[platform]} missing')
                    continue
                run, _, problems, output = p.run_example(platform, path.read_text(), tools)
                if problems or output is None:
                    failures.append(f'{task}/{platform}: ' + '; '.join(problems or ['no output']))
                    continue
                target = p.expected_path(task, platform)
                changed = not target.is_file() or target.read_text() != output
                target.write_text(output)
                print(f"  {task}/{platform}: {'written' if changed else 'unchanged'} ({run['seconds']} s)", flush=True)
    if failures:
        print('Failures:\n  ' + '\n  '.join(failures))
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
