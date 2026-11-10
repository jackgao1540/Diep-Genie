import copy

import pytest

from scripts.plot_tokenizer_continuation import column_limits, plot_data, synthetic_fixture


def test_plot_grid_has_fixed_populations_steps_and_shared_coverage_boundary():
    fixture=synthetic_fixture()
    panels,boundary=plot_data(fixture)
    assert boundary==2270 and len(panels)==6
    assert [(p['split'],p['metric']) for p in panels]==[
        (split,metric) for split in ('train','val') for metric in ('mse','foreground_mse','temporal_mse')]
    assert all(len(panel[arm])==5 for panel in panels for arm in ('baseline','detail'))
    assert panels[0]['baseline'][0]==.01
    assert panels[3]['baseline'][0]==pytest.approx(.012)
    assert fixture['synthetic_plotting_test'] is True
    limits=column_limits(panels)
    for index,panel in enumerate(panels):
        low,high=limits[index%3]
        assert low==0 and high>max(panel['baseline']+panel['detail'])


@pytest.mark.parametrize('fault',['unfinished','coverage','window_count','missing','duplicate','population','nan','gate'])
def test_incomplete_or_inconsistent_final_evidence_cannot_be_plotted(fault):
    fixture=copy.deepcopy(synthetic_fixture())
    if fault=='unfinished':fixture['run_status']='budget_exhausted'
    if fault=='coverage':fixture['coverage']['full_coverage']=False
    if fault=='window_count':fixture['coverage']['coverage_windows']=20332
    if fault=='missing':fixture['paired_assessments'].pop()
    if fault=='duplicate':fixture['paired_assessments'][-1]=copy.deepcopy(fixture['paired_assessments'][0])
    if fault=='population':fixture['paired_assessments'][0]['sources']=83
    if fault=='nan':fixture['paired_assessments'][0]['mse']['pooled']['reference']=float('nan')
    if fault=='gate':fixture['quality_gates']='passed'
    with pytest.raises(ValueError):plot_data(fixture)
