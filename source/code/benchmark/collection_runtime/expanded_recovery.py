"""Versioned native-awk scanner correction and six local execution recoveries."""
from __future__ import annotations
import argparse
import copy
import importlib.util
from ._loader import load_sibling
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[3]
REPORT=ROOT/'quality_reports/reports/2026-09-16_expanded-prompt-evaluation'
WORK=REPORT/'work/recovery'
ORIGINAL=REPORT/'work/collector/applied8_v2/manifest.json'
SOURCE = Path(__file__).with_name('expanded.py')
DECISION=ROOT/'quality_reports/decision_records/2026-09-16_expanded-prompt-runtime-recovery.md'
m = load_sibling(__package__, 'expanded')
c=m.c
ORIGINAL_SCANNER=c.access_findings
BASE_COMPLETED=m.completed_record
BASE_RUN=m.run_job


def corrected_access_findings(text,cwd):
    # Normalize this installed executable only at shell command position.
    pattern=r'(?P<prefix>(?:^|[\n;&|])\s*|(?:-lc|-c)\s+[\"\x27])/usr/bin/awk(?=\s|$)'
    normalized=re.sub(pattern,lambda match:match.group('prefix')+'awk',str(text))
    return ORIGINAL_SCANNER(normalized,cwd)
c.access_findings=corrected_access_findings


def prepare():
    original=json.loads(ORIGINAL.read_text());c.verify_manifest(original)
    root=ORIGINAL.parent/'runs'/original['protocol_sha256'];entries=[]
    pins={str(Path(__file__)):c.base.sha(Path(__file__)),str(SOURCE):c.base.sha(SOURCE),str(DECISION):c.base.sha(DECISION),str(ORIGINAL):c.base.sha(ORIGINAL)}
    for path in sorted(root.glob('*/attempt_*/run_record.json')):
        record=json.loads(path.read_text())
        if record.get('outcome',{}).get('primary_success') is not None:continue
        assert record['provider_ok'] and len(record['calls'])==1
        assert record['access_findings'] or record.get('driver_error')=="RuntimeError('Collection stop requested before subprocess start')"
        for item in path.parent.rglob('*'):
            assert not item.is_symlink()
            if item.is_file():pins[str(item)]=c.base.sha(item)
        entries.append(dict(job=record['job'],original_record=str(path),original_record_sha256=c.base.sha(path)))
    assert len(entries)==6
    protocol=dict(version='native_awk_recovery_v1',original_manifest=str(ORIGINAL),original_protocol_sha256=original['protocol_sha256'],
                  correction='Treat /usr/bin/awk as installed software only at command position. Keep external argument and mixed-command checks.',
                  decision_record=str(DECISION),decision_record_sha256=c.base.sha(DECISION),source_hashes=pins,entries=entries,
                  provider_retries=0,local_recovery='Reexecute unchanged relative-path submissions and retain original records.',
                  future_requests='Same original manifest, jobs, prompts, provider argv, settings, scoring, and deadlines. Record this runtime correction version.')
    manifest=dict(created_at=c.now(),protocol_sha256=c.digest(protocol),protocol=protocol)
    path=WORK/'manifest.json'
    if path.exists() and json.loads(path.read_text())['protocol_sha256']!=manifest['protocol_sha256']:raise ValueError('Preserve existing recovery version')
    c.save(path,manifest);print(json.dumps(dict(recovery_protocol_sha256=manifest['protocol_sha256'],held=6,provider_calls=0)))


def recovery_manifest():
    manifest=json.loads((WORK/'manifest.json').read_text());c.verify_manifest(manifest)
    original=json.loads(ORIGINAL.read_text());c.verify_manifest(original)
    return manifest,original


def recover_local():
    version,original=recovery_manifest();receipts=[]
    assert not c.STOP.is_set(),'Local recovery must run in a separate idle process'
    for entry in version['protocol']['entries']:
        path=Path(entry['original_record']);record=json.loads(path.read_text());job=record['job'];destination=WORK/'records'/c.job_id(job)
        if (destination/'receipt.json').exists():
            receipts.append(json.loads((destination/'receipt.json').read_text()));continue
        if destination.exists():raise ValueError('Preserve incomplete recovery before retry')
        destination.mkdir(parents=True)
        command=json.loads((path.parent/'command_attempt1.json').read_text());old_cwd=Path(command['cwd'])
        assert not c.tool_violations(job,record['calls'],old_cwd)
        submission=path.parent/'submitted_workspace';script=submission/'run.py';text=script.read_text()
        assert str(old_cwd) not in text and not corrected_access_findings(text,old_cwd)
        assert not any(p.is_symlink() for p in submission.rglob('*'))
        spec=original['protocol']['tasks'][job['task']]
        assert c.base.sha(submission/'input.csv')==c.base.sha(Path(spec['data_path']))
        shutil.copytree(submission,destination/'submitted_workspace')
        derived=copy.deepcopy(record)
        for key in ['driver_error','outcome','accounting_observations','inherited_runtime_record']:derived.pop(key,None)
        derived.update(access_findings=[],protocol_violation=False,executions=[],attempt_directory=str(destination))
        with tempfile.TemporaryDirectory(prefix='expanded_local_recovery_',dir='/private/tmp') as temp:
            cwd=Path(temp)
            for item in submission.iterdir():
                if item.name in {'result.json','input.csv','run.log','printed_output.txt'}:continue
                if item.is_dir():shutil.copytree(item,cwd/item.name)
                else:shutil.copy2(item,cwd/item.name)
            shutil.copyfile(spec['data_path'],cwd/'input.csv')
            assert c.base.sha(cwd/'run.py')==c.base.sha(script)
            execution=c.execute(cwd/'run.py','python',cwd,destination,99)
            estimates,rule,notes=c.load_output(cwd,spec,job,execution)
            numeric,terms,numeric_notes=c.compare_numeric(estimates,spec)
            integrity,integrity_notes=c.sp.score_integrity(c.base.as_task(spec),cwd,cwd/'run.py',{spec['task_id']:spec['reference']})
            integrity=bool(integrity and c.base.sha(cwd/'input.csv')==c.base.sha(Path(spec['data_path'])) and c.base.sha(cwd/'run.py')==c.base.sha(script))
            assert integrity,integrity_notes
            c.prior.snapshot(cwd,destination/'final_workspace')
            derived.update(executions=[execution],status='execution_timeout' if execution.get('timed_out') else 'completed',
                           primary_output_path=str(destination/'final_workspace/result.json'),completed_at=c.now())
            derived['scoring']=dict(success=bool(numeric and integrity),numeric_success=numeric,integrity_pass=integrity,estimates=estimates,
                                    per_term=terms,notes=notes+numeric_notes+integrity_notes,primary_source='fresh local execution of preserved submission')
        # The prospective adapter uses an attempt number; this is an execution recovery, not another provider attempt.
        c.save(destination/'recovered_runtime_record.json',derived)
        adapted=m.adapt_record(derived,destination/'attempt_0001')
        adapted['inherited_runtime_record']=str(destination/'recovered_runtime_record.json')
        adapted['original_inherited_runtime_record']=str(path.parent/'inherited_runtime_record.json')
        adapted['recovery']=dict(original_record=str(path),original_record_sha256=entry['original_record_sha256'],recovery_protocol_sha256=version['protocol_sha256'],provider_calls_added=0)
        c.save(destination/'run_record.json',adapted)
        receipt=dict(status='local_execution_recovered',job=job,protocol_sha256=original['protocol_sha256'],recovery_protocol_sha256=version['protocol_sha256'],
                     original_record=str(path),original_record_sha256=entry['original_record_sha256'],recovered_record=str(destination/'run_record.json'),
                     recovered_record_sha256=c.base.sha(destination/'run_record.json'),decision_record=str(DECISION),decision_record_sha256=c.base.sha(DECISION),
                     provider_calls_added=0,unchanged_prompt_argv_data_script=True,primary_success=adapted['outcome']['primary_success'],numerical_success=adapted['outcome']['numerical_success'])
        c.save(destination/'receipt.json',receipt);receipts.append(receipt)
        assert c.base.sha(path)==entry['original_record_sha256']
    recovery_manifest();c.save(WORK/'receipts.json',dict(receipts=receipts,provider_calls_added=0))
    print(json.dumps(dict(recovered=len(receipts),successes=sum(r['primary_success'] for r in receipts),provider_calls_added=0)))


def overlay_completed(folder,protocol_hash):
    base=BASE_COMPLETED(folder,protocol_hash)
    if base is not None:return base
    receipt=WORK/'records'/Path(folder).name/'receipt.json'
    if not receipt.exists():return None
    review=json.loads(receipt.read_text())
    if review.get('status')!='local_execution_recovered':raise ValueError('Unsupported recovery receipt status')
    if review['protocol_sha256']!=protocol_hash:raise ValueError('Recovery protocol mismatch')
    for key,hash_key in [('original_record','original_record_sha256'),('recovered_record','recovered_record_sha256'),('decision_record','decision_record_sha256')]:
        if c.base.sha(Path(review[key]))!=review[hash_key]:raise ValueError('Recovery receipt hash mismatch')
    record=json.loads(Path(review['recovered_record']).read_text())
    if record['job']!=review['job'] or record['protocol_sha256']!=protocol_hash:raise ValueError('Recovered assignment mismatch')
    return record


def corrected_run(job,manifest,run_root,**kwargs):
    version,_=recovery_manifest()
    existing=overlay_completed(Path(run_root)/c.job_id(job),manifest['protocol_sha256'])
    if existing is not None:return existing
    record=BASE_RUN(job,manifest,run_root,**kwargs)
    if record is None:return None
    record['runtime_correction']=dict(recovery_protocol_sha256=version['protocol_sha256'],source=str(Path(__file__)),source_sha256=c.base.sha(Path(__file__)),provider_treatment_unchanged=True)
    c.save(Path(record['attempt_directory'])/'run_record.json',record)
    return record


def main(argv=None):
    args=list(sys.argv[1:] if argv is None else argv)
    if args==['prep']:prepare();return
    if args==['recover']:recover_local();return
    if args==['check']:
        recovery_manifest();m.main(['check','--manifest',str(ORIGINAL)]);return
    version,original=recovery_manifest()
    m.completed_record=overlay_completed;c.completed_record=overlay_completed;m.run_job=corrected_run;c.WORK=WORK
    if '--manifest' not in args:args+=['--manifest',str(ORIGINAL)]
    m.main(args)

if __name__=='__main__':main()
