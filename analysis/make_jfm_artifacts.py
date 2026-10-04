"""Generate the cached-tape analysis tables/figures from a5/a6 JSONs.

Run after a5_cached_benchmark.py and a6_cached_economics.py. Outputs are small,
versioned manuscript artifacts; derived input tapes are never rewritten.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil

from d0_common import HERE, REPO, read_json
from make_tables import _table, fmt, pct, day_label

PAPER = REPO / 'artifacts'
RULES = [('lr', 'Lee--Ready'), ('emo', 'quote rule'), ('tick', 'tick test'), ('bvc_5s', 'BVC, 5 s bars')]


def benchmark_table(data: dict) -> str:
    s = data['summary']['full_history']
    rows = []
    for rule, name in RULES:
        own, common = s['own_sample'][rule], s['common_sample'][rule]
        rows.append(' & '.join([name, fmt(own['balanced_accuracy']['equal_day_mean']),
                              pct(own['coverage_of_matched']['equal_day_mean']),
                              fmt(common['balanced_accuracy']['equal_day_mean']),
                              fmt(common['accuracy']['equal_day_mean']), fmt(common['mcc']['equal_day_mean'])]))
    coverage = [d['full_history']['common_coverage'] for d in data['days']]
    n = sum(d['full_history']['common_rows'] for d in data['days'])
    cap = ('Full cached daily histories, scored against hash-matched taker sides. Cells are equal-day means over twelve selected dates. '
           'Own-sample scores condition on each rule assigning a sign; own coverage is the percentage of matched prints signed. '
           'Common scores use the same intersection of four assigned signs on each day, ' + fmt(n) + ' prints in total, '
           'covering ' + pct(min(coverage)) + '--' + pct(max(coverage)) + '\\% of matched prints by day. '
           'BVC is a retrospective five-second bar-majority diagnostic. Means and ranges describe selected days; no population test is implied.')
    return _table('lrrrrr', 'rule & own BA & own cover (\\%) & common BA & common accuracy & common MCC', rows,
                  cap, 'tab:cached-benchmark')


def history_table(data: dict) -> str:
    rows = []
    for rule, name in RULES:
        ds = [d['paired_history'][rule] for d in data['days']]
        mean = lambda key: sum(d[key] for d in ds) / len(ds)
        flips = [d['sign_changed_share'] for d in ds]
        diffs = [d['balanced_accuracy_difference_full_minus_selected'] for d in ds]
        rows.append(' & '.join([name, fmt(sum(d['full_history']['balanced_accuracy'] for d in ds)/len(ds)),
                              fmt(sum(d['selected_history']['balanced_accuracy'] for d in ds)/len(ds)),
                              fmt(mean('balanced_accuracy_difference_full_minus_selected'), 4),
                              '['+fmt(min(diffs),4)+', '+fmt(max(diffs),4)+']',
                              pct(mean('sign_changed_share'), 3), '['+pct(min(flips),3)+', '+pct(max(flips),3)+']']))
    cap = ('History construction comparison on identical cache row ids where both histories assign the rule a sign. '
           'Full retains all cached prints before matching; selected retains only settlement-matched prints before signing. '
           'BA denotes balanced accuracy; $\\Delta$ is full minus selected. Score columns and flip share are equal-day means; brackets give day ranges. '
           'The paired sample is rule-specific and differs from the four-rule common sample. Signs reset each UTC day. '
           'Changes in abstention are recorded separately in the result JSON.')
    return _table('lrrrrrr', 'rule & full BA & selected BA & $\\Delta$ BA & range $\\Delta$ & flipped (\\%) & range (\\%)',
                  rows, cap, 'tab:cached-history', colsep='3pt')


def cached_days_table(data: dict) -> str:
    rows = []
    for d in data['days']:
        h=d['full_history'];c=h['common_sample']
        rows.append(' & '.join([day_label(d['day']),fmt(d['audit']['matched_prints']),fmt(h['common_rows']),pct(h['common_coverage']),
                              pct(c['lr']['truth_buys']/c['lr']['n']),fmt(c['lr']['balanced_accuracy']),
                              fmt(c['tick']['balanced_accuracy']),fmt(c['bvc_5s']['balanced_accuracy'])]))
    cap = ('Daily denominators and balanced accuracy on the full-history common sample. Buy share is computed on that same sample. '
           'V1 denotes the two first-generation exchange dates. The common intersection is a selected subset of matched prints, '
           'not representative of all venue transactions. All four rules share these observations; the quote rule is tabulated in the JSON.')
    return _table('lrrrrrrr','day & matched & common & cover (\\%) & buys (\\%) & LR BA & tick BA & BVC BA',rows,cap,'tab:cached-days',size='scriptsize')


def write_table(name: str, text: str) -> None:
    path=HERE/f'tab_{name}.tex';path.write_text(text);shutil.copy2(path,PAPER/f'tables_{name}.tex')


def benchmark_figure(data: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from make_figures_v2 import _style, _panel, _publication_fonts, SERIES
    xs=[dt.date.fromisoformat(d['day']) for d in data['days']]
    fig,(ax,bx)=plt.subplots(1,2,figsize=(9.2,3.2),gridspec_kw={'wspace':.3})
    for key,name,color in SERIES:
        ax.plot(xs,[d['full_history']['common_sample'][key]['balanced_accuracy'] for d in data['days']],
                color=color,marker='o',markersize=3,label=name,linestyle='--' if key=='emo' else '-')
        bx.plot(xs,[100*d['paired_history'][key]['balanced_accuracy_difference_full_minus_selected'] for d in data['days']],
                color=color,marker='o',markersize=3,linestyle='--' if key=='emo' else '-')
    for a in (ax,bx):
        _style(a);a.xaxis.set_major_locator(mdates.MonthLocator());a.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.set_ylabel('balanced accuracy, common sample');ax.set_ylim(.45,1);ax.axhline(.5,color='#333333',linestyle=':',linewidth=.7)
    bx.set_ylabel('history BA difference\n(percentage points)');bx.axhline(0,color='#333333',linewidth=.7)
    ax.legend(loc='lower left',ncol=2,fontsize=7);_panel(ax,'a');_panel(bx,'b')
    _publication_fonts(fig,6.27)
    out=HERE/'fig_cached_direction.pdf';fig.savefig(out,bbox_inches='tight');plt.close(fig);shutil.copy2(out,PAPER/'figures'/out.name)


def manuscript_summary(data: dict) -> dict:
    days=data['days'];mean=lambda xs: sum(xs)/len(xs)
    return {
        'cached_prints':sum(d['audit']['cached_prints'] for d in days),
        'matched_prints':sum(d['audit']['matched_prints'] for d in days),
        'common_prints':sum(d['full_history']['common_rows'] for d in days),
        'common_coverage_min':min(d['full_history']['common_coverage'] for d in days),
        'common_coverage_max':max(d['full_history']['common_coverage'] for d in days),
        'matched_tied_prints':sum(d['audit']['matched_tied_prints'] for d in days),
        'matched_tied_share':sum(d['audit']['matched_tied_prints'] for d in days)/sum(d['audit']['matched_prints'] for d in days),
        'untied_common_prints':sum(d['tie_exclusion_full_history']['common_rows'] for d in days),
        'repeated_hash_rows':sum(d['audit']['matched_repeated_hash_rows'] for d in days),
        'untied_common_balanced_accuracy':{r:mean([d['tie_exclusion_full_history']['common_sample'][r]['balanced_accuracy'] for d in days]) for r,_ in RULES},
        'unique_hash_common_balanced_accuracy':{r:mean([d['unique_hash_full_history']['common_sample'][r]['balanced_accuracy'] for d in days]) for r,_ in RULES},
        'paired_history_max_flip_share':{r:max(d['paired_history'][r]['sign_changed_share'] for d in days) for r,_ in RULES},
        'paired_history_total_changed_signs':{r:sum(d['paired_history'][r]['sign_changed'] for d in days) for r,_ in RULES},
        'lr_fixed_lag_balanced_accuracy':{r:mean([d['lr_lag_sensitivity_full_history']['scores'][r]['balanced_accuracy'] for d in days]) for r in days[0]['lr_lag_sensitivity_full_history']['scores']},
        'source_json_sha256':hashlib.sha256((HERE/'a5_cached_benchmark.json').read_bytes()).hexdigest(),
    }


MEASURES = [('effective', 'effective'), ('realised', 'realised'), ('impact', '5-min impact')]
WEIGHTS = [('fill', 'Equal-fill weights'), ('share_volume', 'Share-volume weights')]


def economics_means_table(data: dict) -> str:
    from make_tables import _panel
    panels=[]
    for weight,title in WEIGHTS:
        panel=data['summary']['primary']['weights'][weight];rows=[]
        rows.append(' & '.join(['taker',*[fmt(100*panel['taker'][m]['equal_day_mean'],3) for m,_ in MEASURES]]))
        for rule,name in RULES:
            rows.append(' & '.join([name,*[fmt(100*panel['rules'][rule]['means'][m]['equal_day_mean'],3) for m,_ in MEASURES]]))
        panels.append(_panel('lrrr',title,'sign & effective & realised & 5-min impact',rows))
    caption=('Signed accounting quantities in cents of collateral per share, on the identical primary eligible fills for every rule and the taker. '
             'Cells average the twelve daily means equally. Each daily equal-fill or share-volume weighting is shared across all signs. '
             'Effective equals realised plus five-minute midpoint impact before rounding. The quote is delivered to the collector; '
             'these are not fee-adjusted profits or identified causal impact/information estimates.')
    return ('\\begin{table}[htbp]\n\\centering\\footnotesize\n'+ '\n\\par\\medskip\n'.join(panels)+
            '\n\\caption{'+caption+'}\n\\label{tab:cached-economic-means}\n\\end{table}\n')


def economics_distortion_table(data: dict) -> str:
    from make_tables import _panel
    panels=[]
    for weight,title in WEIGHTS:
        panel=data['summary']['primary']['weights'][weight];rows=[]
        for rule,name in RULES:
            v=panel['rules'][rule]
            rows.append(' & '.join([name,*[fmt(100*v['differences'][m]['equal_day_mean'],3) for m,_ in MEASURES],
                                  *[str(v['sign_reversals'][m]['opposite_sign_days']) for m,_ in MEASURES]]))
        panels.append(_panel('lrrrrrr',title,'rule & $\\Delta E$ & $\\Delta R$ & $\\Delta I$ & flip E & flip R & flip I',rows))
    cap=('Rule minus taker signed means in cents of collateral per share, with the same rows and weights; equal-day means of paired differences. '
         'E, R and I denote effective spread, realised spread and five-minute midpoint impact. Flip columns count dates with opposite daily mean signs (out of twelve); '
         'absolute means at or below $10^{-12}$ price units are treated as numerical zero. Differences preserve direction rather than taking absolute values. '
         'They measure distortion relative to delivered-quote accounting and do not identify changes in market behavior.')
    return ('\\begin{table}[htbp]\n\\centering\\footnotesize\n'+'\n\\par\\medskip\n'.join(panels)+
            '\n\\caption{'+cap+'}\n\\label{tab:cached-economic-distortion}\n\\end{table}\n')


def economic_eligibility_table(data: dict) -> str:
    rows=[]
    for d in data['days']:
        p=d['primary'];steps={x['step']:x['after'] for x in p['funnel']}
        rows.append(' & '.join([day_label(d['day']),fmt(steps['settlement_matched']),
                     fmt(steps['all_four_rule_signs_and_valid_truth']),fmt(steps['strictly_interior_uncrossed_current_quotes']),
                     fmt(p['sample']['rows']),pct(p['sample']['rows']/steps['settlement_matched']),
                     fmt(d['sensitivities']['quote_age_1s']['sample']['rows']),fmt(p['asset_exploratory']['assets'])]))
    cap=('Sequential economic eligibility: settlement-matched rows; all-four signed rows; rows remaining through current interior uncrossed quotes '
         '(after valid print-price screening); final primary rows after forward-midpoint, positive-size and exclusive day-horizon screening. '
         'Fresh rows additionally require a current quote no more than one second old. Assets have at least fifty common eligible fills on that date; '
         'counts are asset-days and do not imply independence. The future quote age and book state are not stored in the cache.')
    return _table('lrrrrrrr','day & matched & signed & interior & primary & cover (\\%) & fresh & assets',rows,cap,
                  'tab:cached-economic-eligibility',size='scriptsize',colsep='3pt')


def economic_ranks_table(data: dict) -> str:
    from make_tables import _panel
    panels=[]
    for weight,title in WEIGHTS:
        panel=data['summary']['asset_exploratory']['weights'][weight];rows=[]
        for rule,name in RULES:
            v=panel[rule]
            rows.append(' & '.join([name,*[fmt(v[m]['spearman_rank_correlation']['equal_day_mean'],3) for m,_ in MEASURES],
                                  *[pct(v[m]['sign_reversal_share']['equal_day_mean'],1) for m,_ in MEASURES]]))
        panels.append(_panel('lrrrrrr',title,'rule & $\\rho_E$ & $\\rho_R$ & $\\rho_I$ & flip E (\\%) & flip R (\\%) & flip I (\\%)',rows))
    cap=('Exploratory per-asset daily mean comparisons on identical eligible fills and assets with at least fifty observations. '
         'The $\\rho$ columns correlate rule-signed and taker-signed asset means: E is effective spread, R realised spread, and I five-minute midpoint impact. '
         'Rank correlations and opposite-sign asset shares are averaged equally across the twelve dates. '
         'Spearman inputs are rounded to twelve decimals to stabilize numerical ties; negligible means use the same $10^{-12}$ zero convention. '
         'Assets can share markets/events; no independent-asset inference or population claim is made. These are numerical zero conventions, not economic materiality cutoffs.')
    return ('\\begin{table}[htbp]\n\\centering\\footnotesize\n'+'\n\\par\\medskip\n'.join(panels)+
            '\n\\caption{'+cap+'}\n\\label{tab:cached-economic-ranks}\n\\end{table}\n')


def economic_sensitivity_table(data: dict) -> str:
    rows=[]
    for key,name in [('primary','primary'),('quote_age_1s','current quote age $\\leq$ 1 s'),('tie_exclusion','untied scored rows'),('unique_hash','one eligible row/hash'),('chain_price','on-chain price')]:
        q=data['summary']['primary'] if key=='primary' else data['summary']['sensitivities'][key]
        v=q['weights']['share_volume'];rs=v['rules']
        rows.append(' & '.join([name,fmt(q['eligible_rows_total']),fmt(100*v['taker']['impact']['equal_day_mean'],3),
                    *[fmt(100*rs[r]['means']['impact']['equal_day_mean'],3) for r in ['lr','tick','bvc_5s']],
                    *[str(rs[r]['sign_reversals']['impact']['opposite_sign_days']) for r in ['lr','tick','bvc_5s']]]))
    cap=('Five-minute midpoint impact under printed-share-volume weights, in cents per share; cells average daily means equally. '
         'Opposite-sign columns count daily rule/taker mean reversals out of twelve dates, with the stated numerical-zero tolerance. '
         'Each row compares all signs on identical observations, but populations differ across sensitivity rows. Signing histories are retained. '
         'The chain-price row holds signs, weights and midpoints fixed, so midpoint impact is invariant to its price substitution. '
         'Untied scoring removes the aggregate tick/BVC sign reversal; it does not establish venue ordering.')
    return _table('lrrrrrrrr','sample & fills & taker & LR & tick & BVC & flip LR & flip tick & flip BVC',rows,cap,
                  'tab:cached-economic-sensitivity',size='scriptsize',colsep='3pt')


def economics_figure(data: dict) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from make_figures_v2 import _style, _panel, _publication_fonts, SERIES
    fig,axes=plt.subplots(1,3,figsize=(9.4,3.2),gridspec_kw={'wspace':.35},sharey=True)
    for ax,(measure,label),letter in zip(axes,MEASURES,'abc'):
        for row,(rule,name,color) in enumerate(SERIES):
            for weight,offset,marker in [('fill',.11,'o'),('share_volume',-.11,'D')]:
                v=data['summary']['primary']['weights'][weight]['rules'][rule]['differences'][measure]
                mean,lo,hi=(100*v[k] for k in ('equal_day_mean','min','max'))
                ax.errorbar(mean,row+offset,xerr=[[mean-lo],[hi-mean]],color=color,marker=marker,markersize=4,
                            linestyle='none',capsize=2,linewidth=.8,label='equal-fill' if weight=='fill' else 'share-volume')
        _style(ax);ax.axvline(0,color='#333333',linewidth=.7);ax.set_xlabel('rule minus taker\n(cents/share)')
        _panel(ax,letter);ax.text(.5,1.02,label,transform=ax.transAxes,ha='center',va='bottom')
    axes[0].set_yticks(range(4));axes[0].set_yticklabels([name for _,name,_ in SERIES]);axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    axes[1].legend(handles=[Line2D([],[],color='#333333',marker='o',linestyle='none',label='equal-fill'),
                           Line2D([],[],color='#333333',marker='D',linestyle='none',label='share-volume')],
                   loc='upper center',bbox_to_anchor=(.5,-.38),ncol=2,fontsize=7)
    _publication_fonts(fig,6.27)
    out=HERE/'fig_cached_economics.pdf';fig.savefig(out,bbox_inches='tight');plt.close(fig);shutil.copy2(out,PAPER/'figures'/out.name)


def main() -> None:
    data=read_json('a5_cached_benchmark')
    summary={'classifier':manuscript_summary(data)}
    write_table('cached_benchmark',benchmark_table(data))
    write_table('cached_history',history_table(data))
    write_table('cached_days',cached_days_table(data))
    benchmark_figure(data)
    if (HERE/'a6_cached_economics.json').exists():
        econ=read_json('a6_cached_economics')
        summary['economics']={'eligible_rows_total':econ['summary']['primary']['eligible_rows_total'],
                              'coverage_of_matched':econ['summary']['primary']['eligible_rows_total']/summary['classifier']['matched_prints'],
                              'source_json_sha256':hashlib.sha256((HERE/'a6_cached_economics.json').read_bytes()).hexdigest()}
        write_table('cached_economic_means',economics_means_table(econ))
        write_table('cached_economic_distortion',economics_distortion_table(econ))
        write_table('cached_economic_eligibility',economic_eligibility_table(econ))
        write_table('cached_economic_ranks',economic_ranks_table(econ))
        write_table('cached_economic_sensitivity',economic_sensitivity_table(econ))
        economics_figure(econ)
    (HERE/'jfm_revision_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
