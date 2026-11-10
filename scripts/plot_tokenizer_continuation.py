"""Private learning curves from an audit-bound completed continuation report.

The optional synthetic plotting fixture is prominently labeled and cannot be
mistaken for an audited model result. This tool changes no training artifacts.
"""
import argparse
import json
import math
from pathlib import Path

from scripts.summarize_tokenizer_continuation import METRICS, require, sha, summarize


STEPS = [999, 2000, 3000, 4000, 5000]
TITLES = ['Whole-frame error', 'Colored-foreground error', 'Temporal-difference error']
SPLITS = [('train', 84, 84), ('val', 64, 16)]


def plot_data(summary):
    require(summary['run_status'] == 'completed' and summary['latest_common_step'] == 5000,
            'Final learning curves require the completed matched step5000 run')
    require(summary['quality_gates'] == 'unmeasured' and summary['promotion_approved'] is False,
            'Learning curves cannot award a model-quality gate')
    coverage = summary['coverage']
    require(coverage['full_coverage'] is True and coverage['coverage_unique_frames'] == 315320
            and coverage['coverage_windows'] == 20331, 'Complete frozen TRAIN-frame coverage is required')
    coverage_step = 999+math.ceil(coverage['coverage_windows']/16)
    require(coverage_step == 2270, 'Unexpected full-coverage boundary')
    entries = {(row['step'], row['split']):row for row in summary['paired_assessments']}
    require(len(entries) == len(summary['paired_assessments']) == 10
            and set(entries) == {(step, split) for step in STEPS for split, _, _ in SPLITS},
            'Require all five complete paired train/validation assessments')
    panels = []
    for split, windows, sources in SPLITS:
        for metric in METRICS:
            baseline, detail = [], []
            for step in STEPS:
                row = entries[step, split]
                require(row['samples'] == windows and row['sources'] == sources,
                        'Fixed probe population changed')
                values = row[metric]['pooled']
                for name, target in [('reference', baseline), ('candidate', detail)]:
                    value = values[name]
                    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
                            'A plotted metric is missing or invalid')
                    target.append(value)
            panels.append(dict(split=split, metric=metric, baseline=baseline, detail=detail))
    return panels, coverage_step


def synthetic_fixture():
    """Drawing verification only: no real checkpoint measurements are embedded."""
    rows = []
    for split, windows, sources in SPLITS:
        for index, step in enumerate(STEPS):
            row = dict(step=step, split=split, samples=windows, sources=sources)
            for metric, scale in zip(METRICS, [.01, .03, .008]):
                baseline = scale*[1., .72, .56, .48, .43][index]*(1. if split == 'train' else 1.2)
                detail = baseline*[.95, .99, .88, .91, .85][index]
                row[metric] = dict(pooled=dict(reference=baseline, candidate=detail))
            rows.append(row)
    return dict(run_status='completed', latest_common_step=5000, quality_gates='unmeasured',
                promotion_approved=False, paired_assessments=rows,
                coverage=dict(full_coverage=True,coverage_unique_frames=315320,coverage_windows=20331),
                synthetic_plotting_test=True)


def column_limits(panels):
    """Shared row scales must include both populations, even if validation is larger."""
    result=[]
    for column in range(3):
        maximum=max(value for index in (column,column+3) for arm in ('baseline','detail')
                    for value in panels[index][arm])
        result.append((0.,maximum*1.08 if maximum else 1.))
    return result


def render(summary, output, *, synthetic=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    require(bool(summary.get('synthetic_plotting_test')) == synthetic, 'Synthetic plotting mode mismatch')
    panels, coverage_step = plot_data(summary)
    output = Path(output)
    require(not output.exists(), 'Preserve existing learning-curve figures')
    fig, axes = plt.subplots(2, 3, figsize=(13.5, 8.2), sharex=True, sharey='col')
    fig.subplots_adjust(left=.105, right=.985, bottom=.21, top=.81, hspace=.19, wspace=.19)
    colors = {'baseline':'#1766a6', 'detail':'#d95a13'}
    for panel, ax in zip(panels, axes.flat):
        for arm in ('baseline', 'detail'):
            ax.plot(STEPS, panel[arm], color=colors[arm], marker='o', markersize=4.5, linewidth=1.8)
        ax.axvline(coverage_step, color='#68717a', linestyle='--', linewidth=1.2, zorder=0)
        ax.set_xlim(850, 5150)
        ax.set_xticks(STEPS, ['999', '2,000', '3,000', '4,000', '5,000'])
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(axis='y', alpha=.22, linewidth=.7)
        ax.spines[['top', 'right']].set_visible(False)
        ax.ticklabel_format(axis='y', style='sci', scilimits=(-3, 3), useOffset=False)
    for ax, title in zip(axes[0], TITLES):
        ax.set_title(title, fontsize=12, pad=11)
    limits=column_limits(panels)
    for column, limit in enumerate(limits):
        axes[0,column].set_ylim(*limit)
    axes[0,0].set_ylabel('TRAINING PROBES\n84 windows / 84 sources\nMean squared error', fontsize=10)
    axes[1,0].set_ylabel('VALIDATION PROBES\n64 windows / 16 other sources\nMean squared error', fontsize=10)
    for ax in axes[1]:
        ax.set_xlabel('Total unique optimizer updates per arm', fontsize=10, labelpad=8)
    title = 'SYNTHETIC PLOTTING TEST — NOT MODEL RESULTS' if synthetic else 'Tokenizer learning curves · matched continuation to step 5,000'
    fig.suptitle(title, fontsize=16, weight='bold', y=.975, color='#8b2727' if synthetic else '#17232d')
    fig.text(.5, .927, 'Same paired initialization and coverage schedule · lower reconstruction error is better',
             ha='center', fontsize=11, color='#49545e')
    legend = [Line2D([0],[0],color=colors['baseline'],marker='o',label='Baseline'),
              Line2D([0],[0],color=colors['detail'],marker='o',label='Detail loss'),
              Line2D([0],[0],color='#68717a',linestyle='--',label='Full TRAIN-frame coverage · step 2,270')]
    fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.905),ncol=3,frameon=False,fontsize=10.5)
    fig.text(.105,.12, 'Different train/validation source populations: their separation does not isolate overfitting or data sufficiency.\n'
             'Colored foreground is a fixed source-only mask. Temporal error compares reconstructed frame differences, not predictions.\n'
             'One seed; five fixed measurement points. Full prepared-frame coverage does not establish convergence, geometry fidelity or control.',
             va='top',fontsize=9.5,color='#49545e',linespacing=1.7)
    if synthetic:
        fig.text(.5,.49,'SYNTHETIC TEST',ha='center',va='center',fontsize=45,color='#b43535',alpha=.12,rotation=12)
    try:
        with output.open('xb') as stream:
            fig.savefig(stream,format='png',dpi=180,facecolor='white',metadata={'Software':'Matplotlib '+matplotlib.__version__})
    finally:
        plt.close(fig)
    return dict(matplotlib_version=matplotlib.__version__,dpi=180,width_pixels=2430,height_pixels=1476,
                marker_step=coverage_step,column_y_limits=limits,synthetic_plotting_test=synthetic)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True)
    parser.add_argument('--report');parser.add_argument('--audit')
    parser.add_argument('--out',required=True)
    parser.add_argument('--test-fixture',action='store_true')
    args=parser.parse_args()
    root=Path(args.root).resolve();output=(root/args.out).resolve();receipt_path=output.with_suffix('.receipt.json')
    require(output.suffix=='.png' and output.is_relative_to(root/'runs'), 'Keep scientific figures under private project runs')
    require(not output.exists() and not receipt_path.exists(), 'Preserve existing plot artifacts')
    if args.test_fixture:
        require(not args.report and not args.audit, 'Synthetic fixture cannot use a real report')
        summary=synthetic_fixture()
    else:
        require(args.report and args.audit, 'Final figure requires a bound report and artifact audit')
        require(not output.is_relative_to(Path(args.report).resolve().parent), 'Plot must be outside the audited run')
        summary=summarize(args.report,args.audit)
    output.parent.mkdir(parents=True,exist_ok=True)
    receipt=render(summary,output,synthetic=args.test_fixture)
    receipt.update(plot_sha256=sha(output),plotter_sha256=sha(__file__),optimizer_updates=0,inference_performed=False,
                   quality_gates='unmeasured',promotion_approved=False,
                   input_report_sha256=summary.get('report_sha256'),input_audit_sha256=summary.get('artifact_audit_sha256'))
    with receipt_path.open('x',encoding='utf-8') as stream:
        json.dump(receipt,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
