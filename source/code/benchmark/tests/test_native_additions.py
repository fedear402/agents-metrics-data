"""Only genuinely additive R package metadata can pass the scoped policy."""
from pathlib import Path
import sys,copy
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from prospective_collect_unlimited_additions import additive_packages
def environment():
    return {'r':{'executable':'R','version_and_packages':'R version 4.6.0\n"Package"\t"Version"\t"LibPath"\t"Built"\n"stats"\t"4.6.0"\t"/base"\t"built"'},'stata':{'hash':'fixed'}}
def test_addition_is_recorded():
    old=environment();new=copy.deepcopy(old)
    new['r']['version_and_packages']+='\n"lpSolve"\t"5.6.23"\t"/user"\t"built"'
    assert additive_packages(old,new)[0]['package']=='lpSolve'
@pytest.mark.parametrize('change',['upgrade','remove','shadow','interpreter','stata'])
def test_changes_still_fail(change):
    old=environment();new=copy.deepcopy(old)
    if change=='upgrade':new['r']['version_and_packages']=new['r']['version_and_packages'].replace('"4.6.0"','"4.6.1"')
    if change=='remove':new['r']['version_and_packages']='\n'.join(new['r']['version_and_packages'].splitlines()[:2])
    if change=='shadow':new['r']['version_and_packages']+='\n"stats"\t"4.6.0"\t"/user"\t"built"'
    if change=='interpreter':new['r']['executable']='another R'
    if change=='stata':new['stata']['hash']='changed'
    with pytest.raises(ValueError):additive_packages(old,new)
