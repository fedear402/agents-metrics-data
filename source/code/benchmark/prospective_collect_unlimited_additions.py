"""Permit recorded R package additions in authorized unlimited collections.

Existing package metadata and interpreter identities must remain unchanged.
Stata keeps its exact environment check. Source, input, and reference checks
remain the central collector's checks.
"""
from pathlib import Path
import csv
import io
import json
import tempfile
import prospective_collect_unlimited as unlimited

central=unlimited.central


def package_rows(text):
    lines=text.splitlines()
    if len(lines)<2:raise ValueError('Missing R package metadata')
    rows=list(csv.reader(lines[1:],delimiter='\t'))
    header=rows.pop(0)
    if header!=['Package','Version','LibPath','Built']:raise ValueError('Unexpected R package metadata columns')
    items={}
    for row in rows:
        if len(row)!=4 or tuple(row[:1]+row[2:3]) in items:raise ValueError('Ambiguous R package metadata')
        items[(row[0],row[2])]=row
    return lines[0],items


def additive_packages(expected,current):
    """Return new R packages while rejecting changed or removed existing rows."""
    if expected.keys()!=current.keys():raise ValueError('Native platforms changed')
    additions=[]
    for platform,old in expected.items():
        new=current[platform]
        if platform!='r':
            if new!=old:raise ValueError('Native environment changed: '+platform)
            continue
        if old.keys()!=new.keys():raise ValueError('R environment fields changed')
        for key in old:
            if key!='version_and_packages' and old[key]!=new[key]:raise ValueError('R interpreter identity changed')
        version,before=package_rows(old['version_and_packages'])
        current_version,after=package_rows(new['version_and_packages'])
        if version!=current_version:raise ValueError('R version changed')
        for key,row in before.items():
            if after.get(key)!=row:raise ValueError('Existing R package changed or disappeared: '+key[0])
        names={key[0] for key in before}
        for key,row in after.items():
            if key in before:continue
            if key[0] in names:raise ValueError('New library can shadow existing R package: '+key[0])
            additions.append(dict(zip(['package','version','library','built'],row)))
    return additions


def verify_native_environment(expected,runtime,jobs):
    current=central.native_environment(runtime,jobs)
    additions=additive_packages(expected,current)
    if additions:
        path=Path(runtime.c.WORK)/'native_environment_observations.jsonl'
        evidence=dict(at=central.now(),policy='record_additive_r_packages_v1',added_r_packages=additions,
                      expected_sha256=central.digest(expected),observed_sha256=central.digest(current),current=current)
        with path.open('a') as stream:stream.write(json.dumps(evidence)+'\n')


BASE_PREPARE=central.prepare
def prepare(config_path,manifest_path):
    target=Path(manifest_path).resolve();target.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.r_addition_prep_',dir=target.parent) as temporary:
        result=BASE_PREPARE(config_path,Path(temporary)/'manifest.json')
    result['protocol']['version']='prospective_unlimited_additive_r_2026_09_20_v1'
    result['protocol']['policy']['native_environment_policy']='Record newly named R packages. Require unchanged existing package rows, interpreter identities, and exact Stata metadata.'
    test=Path(__file__).with_name('tests')/'test_native_additions.py'
    result['protocol']['source_hashes'][str(test)]=central.sha(test)
    result['protocol_sha256']=central.digest(result['protocol'])
    if target.exists():
        old=json.loads(target.read_text())
        if old['protocol_sha256']!=result['protocol_sha256']:raise ValueError('Preserve the existing manifest; select a new destination')
        return old
    central.save(target,result)
    return result


central.verify_native_environment=verify_native_environment
central.prepare=prepare

if __name__=='__main__':central.main()
