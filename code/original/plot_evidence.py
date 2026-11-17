"""Publication figures from frozen evidence. No experiment/model calls.

Means get marginal percentile bootstrap CIs, never paired-difference CIs.
Frozen paired tests and Holm p-values are reported separately. CSFCube
resamples whole queries and recomputes equal-facet macro means; LOO values
are influence diagnostics, not confidence intervals. See figures/README.md.
"""
import hashlib
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/private/tmp/ispa-mpl-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figures'
SEED = 20260929
B = 20000
BLUE0, BLUE1, BLUE, ORANGE, PEACH, GREEN = '#8FAFC2', '#6B92A8', '#1F5F80', '#B95739', '#D4826A', '#4E7A30'
plt.rcParams.update({
    'font.family':'serif', 'font.serif':['Times New Roman','DejaVu Serif'],
    'font.size':8, 'axes.titlesize':9, 'axes.labelsize':8,
    'xtick.labelsize':7.5, 'ytick.labelsize':7.5, 'legend.fontsize':7,
    'axes.spines.top':False, 'axes.spines.right':False, 'axes.linewidth':.6,
    'grid.color':'#e4e7e9', 'grid.linewidth':.5,
    'svg.fonttype':'none', 'pdf.fonttype':42, 'ps.fonttype':42,
    'mathtext.fontset':'stix', 'savefig.facecolor':'white',
})
FILES = ['P0_selection.json','P0_grid.json','P0_external.json','P0_external_inference.json','P1_analysis.json']
data = {n:json.loads((ROOT/n).read_text()) for n in FILES}
audit = {'seed':SEED, 'resamples':B, 'source_sha256':{n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in FILES}, 'selection':{}, 'presentation':{}, 'external':{}}

def ci(values):
    values=np.asarray([v for v in values if v is not None], dtype=float)
    assert np.isfinite(values).all()
    rng=np.random.default_rng(SEED)
    means=values[rng.integers(len(values),size=(B,len(values)))].mean(1)
    return np.quantile(means,[.025,.975])

def axes_style(ax):
    ax.set_axisbelow(True); ax.grid(axis='y')
    ax.tick_params(length=2.5, width=.6)

def save(fig, stem):
    for ext in ['pdf','svg','png']:
        fig.savefig(OUT/f'{stem}.{ext}', dpi=400, metadata={'Creator':'Evidence plotting script'} if ext=='pdf' else None)
    plt.close(fig)

def layout(width, height, n, note_height=.22):
    fig=plt.figure(figsize=(width,height),layout='constrained')
    gs=fig.add_gridspec(2,n,height_ratios=[1,note_height])
    axs=[fig.add_subplot(gs[0,i]) for i in range(n)]
    note=fig.add_subplot(gs[1,:]);note.axis('off')
    return fig,axs,note

def notes(ax,text,size=7):
    ax.text(0,1,text,transform=ax.transAxes,ha='left',va='top',fontsize=size,linespacing=1.3,color='#3d4448')

def bracket(ax,a,b,y,text,h=.012,fontsize=7):
    ax.plot([a,a,b,b],[y,y+h,y+h,y],color='#3d4448',lw=.65,clip_on=False)
    ax.text((a+b)/2,y+h*1.3,text,ha='center',va='bottom',fontsize=fontsize,linespacing=1.1)

def bars_ci(ax,vals,intervals,colors,width=.62,inside=False):
    x=np.arange(len(vals)); vals=np.asarray(vals); intervals=np.asarray(intervals)
    ax.bar(x,vals,color=colors,width=width,edgecolor='white',linewidth=.5,zorder=2)
    ax.errorbar(x,vals,yerr=np.stack([vals-intervals[:,0],intervals[:,1]-vals]),fmt='none',color='#30363a',lw=.75,capsize=2,zorder=3)
    span=ax.get_ylim()[1]-ax.get_ylim()[0]
    for i,v in enumerate(vals):
        yy=v-.04*span if inside else intervals[i,1]+.018*span
        ax.text(i,yy,f'{v:.3f}',ha='center',va='top' if inside else 'bottom',fontsize=7.5,color='white' if inside else '#21282c',fontweight='bold' if inside else 'normal')
    axes_style(ax)

# 1. Selection controls: values and per-system intervals from saved query rows.
s=data['P0_selection.json']; names=['specter_flat','smartfilter','construction_top_m','rolecond_full']
sel_ci={}
for m in ['ndcg','grade','recall']:
    sel_ci[m]=[]
    for n in names:
        v=[r[m] for _,r in sorted(s['per_query'][n].items()) if r[m] is not None]
        assert np.isclose(np.mean(v),s['summary'][n][m])
        c=ci(v);sel_ci[m].append(c)
        audit['selection'][f'{n}:{m}']={'mean':float(np.mean(v)),'ci95':c.tolist(),'n':len(v)}

def selection(compact=False):
    fig,axs,note=layout(7.16,2.65 if compact else 3.4,3,.29 if compact else .24)
    meta=[('ndcg','(a) Pool-normalized NDCG@8',(.88,1.015)),('grade','(b) Returned-paper mean grade',(1.45,2.22)),('recall','(c) Positive recall@8',(.88,1.035))]
    for ax,(m,title,lim) in zip(axs,meta):
        vals=[s['summary'][n][m] for n in names]; ax.set_ylim(*lim)
        bars_ci(ax,vals,sel_ci[m],[BLUE0,BLUE1,BLUE,ORANGE])
        ax.set_xticks(range(4),['Top-8','Smart-\nFilter','Matched\n$m$','RoleCond'],fontsize=7)
        ax.set_title(title,fontsize=8.5,pad=15)
        ax.yaxis.set_major_locator(MultipleLocator(.05 if m=='ndcg' else .2 if m=='grade' else .05))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f' if m!='grade' else '%.1f'))
        # Explicit zoom cue at the bottom of the y spine.
        ax.plot([-.018,.018],[-.015,.015],transform=ax.transAxes,color='#333',lw=.7,clip_on=False)
        if m=='grade': bracket(ax,2,3,2.15,'*** $p$ < 0.001',h=.010)
        elif m=='recall': bracket(ax,2,3,1.012,'*** $p$ < 0.001',h=.002)
        else:
            ax.scatter([3.40],[s['summary']['rolecond_original_order']['ndcg']],marker='D',s=20,color=PEACH,edgecolor='#333',lw=.5,zorder=4,alpha=.8)
            ax.text(.5,1.005,'Matched comparison: Holm $p$ = 0.075',transform=ax.transAxes,ha='center',fontsize=6.8)
    notes(note,'90 queries (recall: 89); 3–8 papers per pool. Top-8 and matched $m$: construction-order controls, not cosine ranking.\n'
        'Means with marginal 95% bootstrap CIs; paired contrasts and Holm-adjusted tests are reported in Table II.\n'
        'Returned counts: 7.29 / 6.79 / 6.06 / 6.06. Diamond: same-set original order; reordering $\\Delta$ = +0.005 [−0.006, +0.016].',6.8)
    save(fig,'selection_controls_compact' if compact else 'selection_controls')
selection();selection(True)

# Column reflow for the six-page manuscript: horizontal bars avoid tiny type.
fig,axs=plt.subplots(3,1,figsize=(3.5,2.65),layout='constrained')
for ax,m,title,lim in zip(axs,['ndcg','grade','recall'],
        ['(a) Pool-normalized NDCG@8','(b) Returned-paper mean grade','(c) Positive recall@8'],
        [(.88,1.025),(1.45,2.2),(.88,1.05)]):
    vals=np.array([s['summary'][n][m] for n in names]); bounds=np.asarray(sel_ci[m]); y=np.arange(4)
    ax.barh(y,vals,color=[BLUE0,BLUE1,BLUE,ORANGE],height=.6,edgecolor='white',lw=.4)
    ax.errorbar(vals,y,xerr=np.stack([vals-bounds[:,0],bounds[:,1]-vals]),fmt='none',color='#30363a',capsize=1.5,lw=.65)
    for i,v in enumerate(vals):
        ax.text(lim[1],i,f'{v:.3f}',fontsize=6.8,va='center',ha='right',bbox={'facecolor':'white','edgecolor':'none','pad':.8})
    ax.set_yticks(y,['Top-8','SmartFilter','Matched $m$','RoleCond'],fontsize=6.8);ax.invert_yaxis()
    ax.set_xlim(*lim);ax.set_title(title,loc='left',fontsize=8,pad=3)
    ax.tick_params(length=2,labelsize=6.5);ax.grid(axis='x',color='#e4e7e9');ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(.05 if m!='grade' else .2))
    ax.text(1.0,1.035,'Holm $p$ = 0.075' if m=='ndcg' else '*** $p$ < 0.001',transform=ax.transAxes,ha='right',fontsize=6.3)
    ax.plot([-.012,.012],[-.02,.02],transform=ax.transAxes,color='#333',lw=.7,clip_on=False)
    if m=='ndcg': ax.scatter(s['summary']['rolecond_original_order']['ndcg'],3.38,marker='D',s=8,color=PEACH,edgecolor='#333',lw=.4,zorder=4)
save(fig,'selection_controls_column')

# 2. Order sweep. The enumeration is fixed; no outcome-dependent ordering.
grid=data['P0_grid.json']['configs']
def order(compact=False):
    fig,axs,note=layout(3.5 if compact else 7.16,1.9 if compact else 3.2,2,.11 if compact else .25)
    for strat,label,color,marker,ls in [('fixed','Fixed',ORANGE,'o','-'),('corrected_proportional','Corrected proportional',BLUE,'s','-'),('two_stage','Two-stage',GREEN,'^','--')]:
        rows=sorted([r for r in grid if r['k']==3 and r['strategy']==strat],key=lambda r:r['order_index'])
        assert [r['order_index'] for r in rows]==list(range(24))
        for ax,m in zip(axs,['ndcg','nonempty_roles']):
            ax.plot(np.arange(1,25),[r[m] for r in rows],label=label,color=color,marker=marker,ls=ls,lw=1 if compact else 1.2,ms=2.5 if compact else 4,mec='white',mew=.4)
    for ax,title,lim in zip(axs,['(a) NDCG@8','(b) Nonempty roles'],[(.75,1),(1.5,4.2)]):
        ax.axvspan(.5,1.5,color='#edf0f2',zorder=0);ax.axvline(1,color='#687780',lw=.7,ls='--',zorder=1)
        ax.set_title(title,fontsize=8 if compact else 9);ax.set_ylim(*lim);ax.set_xlim(.5,24.5)
        ax.set_xticks([1,8,16,24] if compact else [1,4,8,12,16,20,24]);ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.set_xlabel('Role-order index',fontsize=7 if compact else 8);axes_style(ax)
    handles,labels=axs[0].get_legend_handles_labels()
    if compact:
        note.legend(handles,labels,loc='upper left',bbox_to_anchor=(-.025,1.08),ncol=3,columnspacing=.7,handlelength=1.2,handletextpad=.3,frameon=False,fontsize=6)

    else:
        axs[0].legend(loc='lower left',framealpha=.95,edgecolor='#cccccc',fontsize=7)
        notes(note,'Fixed grade across 24 orders: 1.641–1.867 (span 0.226). Each point aggregates 90 queries ($k_r$ = 3).\n'
            'Capacity 12 never binds (pools ≤8). Dashed vertical line: canonical order (closely related → methods → background → cross-field).\n'
            'The fixed enumeration diagnoses quota and deduplication behavior; no best order is selected for deployment.',7)
    save(fig,'order_sensitivity_column' if compact else 'order_sensitivity')
order();order(True)

# 3. Presentation: validity is observed, coverage CI is bootstrapped by query.
p=data['P1_analysis.json']; conds=['F','G','R','W']; cov=[];cov_ci=[]
for c in conds:
    rows=sorted([r for r in p['per_job'] if r['model']=='gpt-5.5' and r['condition']==c and r['valid']],key=lambda r:r['query_id'])
    v=[r['input_coverage'] for r in rows];assert len(v)==36
    assert np.isclose(np.mean(v),p['summary'][f'gpt-5.5:{c}']['input_coverage'])
    cov.append(float(np.mean(v)));cov_ci.append(ci(v));audit['presentation'][c]={'mean':cov[-1],'ci95':cov_ci[-1].tolist(),'n':len(v)}

def presentation(compact=False):
    fig,axs,note=layout(3.5 if compact else 7.16,1.95 if compact else 3.4,2,.01 if compact else .29)
    ax=axs[0]; valid=[p['summary'][f'claude-sonnet-4-6:{c}']['valid_rate'] for c in conds]
    ax.bar(np.arange(4),valid,color='#E8B0A0',width=.62,edgecolor='white',lw=.5,zorder=2)
    ax.axhline(1,color=BLUE,lw=1.2,ls='--')
    ax.text(.5,.91,'GPT-5.5\n144/144 valid' if compact else 'GPT-5.5: 144/144 strictly valid',transform=ax.transAxes,ha='center',va='top',fontsize=7,color=BLUE)
    ax.text(.5,.4,'Claude Sonnet 4.6\n3/144 valid',transform=ax.transAxes,ha='center',fontsize=7,color='#833f2b')
    for i,v in enumerate(valid): ax.text(i,v+.04,f'{round(v*36)}/36',ha='center',fontsize=6.5 if compact else 7.5)
    ax.set_ylim(0,1.08);ax.set_title('(a) Strict validity',fontsize=8 if compact else 9);ax.set_ylabel('Valid fraction',fontsize=7)
    ax.set_xticks(range(4),conds);axes_style(ax)
    ax=axs[1]; ax.set_ylim(0,1.43)
    bars_ci(ax,cov,cov_ci,[BLUE0,'#4E7FA0',ORANGE,PEACH],inside=False)
    ax.set_xticks(range(4),conds);ax.set_title('(b) GPT-5.5 coverage',fontsize=8 if compact else 9)
    ax.set_yticks([0,.5,1.0]);ax.set_ylabel('Input citation coverage',fontsize=7)
    bracket(ax,2,3,1.13,'n.s.',h=.025,fontsize=6.5 if compact else 7)
    bracket(ax,1,2,1.28,'***',h=.025,fontsize=8)
    notes(note,'F: flat; G: neutral groups; R: roles; W: wrong roles.\n'
        'R−G: +0.099 [0.051, 0.149]; Holm $p$ < 0.001.\n'
        'R−W: −0.001 [−0.023, +0.022]; $p$ = 0.968 (aux.).\n'
        'Bars: means with marginal 95% bootstrap CIs ($n$ = 36).\n'
        'Paired CIs in brackets. Coverage is not correctness.' if compact else
        'All 288 calls returned. F: flat; G: neutral groups; R: role groups; W: wrong role groups. GPT-5.5: $n$ = 36 per condition.\n'
        'R−G: +0.099 [0.051, 0.149], Holm $p$ < 0.001; R−W: −0.001 [−0.023, +0.022], auxiliary $p$ = 0.968 (uncorrected).\n'
        'Bars: marginal 95% bootstrap CIs; brackets: paired 95% CIs. A nonsignificant R−W contrast does not establish equivalence.\n'
        'Coverage measures the fraction of input papers cited, not citation correctness or human utility.',6.5 if compact else 7)
    if compact:
        for text in note.texts: text.set_visible(False)
    save(fig,'presentation_intervention_column' if compact else 'presentation_intervention')
presentation();presentation(True)

# 4. External comparison, preserving equal-facet weighting under query resampling.
e=data['P0_external.json']['20260901']; inference=data['P0_external_inference.json']
qids=sorted({r['query_id'] for r in e['specter_score']['per_query_facet'].values()}); facets=['background','method','result']
assert len(qids)==34
rng=np.random.default_rng(SEED); samples=rng.integers(len(qids),size=(B,len(qids)))
ext_ci={}
for system in ['specter_score','heuristic_score']:
    for m in ['ndcg','ap']:
        numer=np.zeros((34,3));denom=np.zeros((34,3))
        for r in e[system]['per_query_facet'].values():
            i,j=qids.index(r['query_id']),facets.index(r['facet']);numer[i,j]+=r[m];denom[i,j]+=1
        den=denom[samples].sum(1);num=numer[samples].sum(1);ok=(den>0).all(1)
        boots=(num[ok]/den[ok]).mean(1);lohi=np.quantile(boots,[.025,.975]);ext_ci[system,m]=lohi
        mean=float((numer.sum(0)/denom.sum(0)).mean());assert np.isclose(mean,e[system]['summary']['macro'][m])
        audit['external'][f'{system}:{m}']={'mean':mean,'ci95':lohi.tolist(),'query_clusters':34,'estimand':'equal-facet macro mean'}
fig,axs,note=layout(3.5,2.45,2,.5)
for ax,m,title in zip(axs,['ndcg','ap'],['(a) NDCG@10','(b) Average precision']):
    vals=[e[n]['summary']['macro'][m] for n in ['specter_score','heuristic_score']]
    assert np.isclose(vals[1]-vals[0],inference[m]['delta'])
    ax.set_ylim(0,.8);bars_ci(ax,vals,[ext_ci[n,m] for n in ['specter_score','heuristic_score']],[BLUE,ORANGE])
    ax.set_xticks([0,1],['SPECTER-\nflat','Text-adapted\nheuristic'],fontsize=6.7);ax.set_title(title,fontsize=8)
    bracket(ax,0,1,.7,'$p$ = 0.003' if m=='ndcg' else '$p$ < 0.001',h=.015,fontsize=6.5)
notes(note,'Heuristic − SPECTER (paired query-cluster 95% CIs):\n'
    'NDCG: −0.101 [−0.157, −0.047]; AP: −0.119 [−0.175, −0.065].\n'
    '34 queries / 50 facets; equal-facet means; zero training.\n'
    'Bars: marginal query-cluster 95% CIs; Holm-adjusted tests.',6.3)
save(fig,'csfcube_diagnostic')
# Reuse the vector master without its separate explanatory-note area.
from matplotlib.backends.backend_pdf import PdfPages
# Re-render the same axes at column height; no image cropping or rasterization.
fig,axs=plt.subplots(1,2,figsize=(3.5,1.75),layout='constrained')
for ax,m,title in zip(axs,['ndcg','ap'],['(a) NDCG@10','(b) Average precision']):
    vals=[e[n]['summary']['macro'][m] for n in ['specter_score','heuristic_score']]
    ax.set_ylim(0,.8);bars_ci(ax,vals,[ext_ci[n,m] for n in ['specter_score','heuristic_score']],[BLUE,ORANGE])
    ax.set_xticks([0,1],['SPECTER-\nflat','Text-adapted\nheuristic'],fontsize=6.7);ax.set_title(title,fontsize=8)
    bracket(ax,0,1,.7,'$p$ = 0.003' if m=='ndcg' else '$p$ < 0.001',h=.015,fontsize=6.5)
save(fig,'csfcube_diagnostic_column')
(OUT/'figure_statistics.json').write_text(json.dumps(audit,indent=2)+'\n')
(OUT/'README.md').write_text('''# Evidence figures — 2026-09-29

Four redesigned figures from frozen P0/P1 evidence, following the four supplied specs.
PDF (embedded TrueType fonts), SVG (editable text), and 400-dpi PNG are exported.
The main paper uses vector PDF. Wide masters follow the 7.16-inch specs;
compact/column variants retain readable fonts at their actual printed size.
The existing pipeline diagram remains Figure 1; LaTeX numbers the four evidence figures automatically.

## Statistical corrections to the specs

- Bars show per-system/per-condition means and **marginal percentile bootstrap 95% CIs**.
  Paired-difference CIs are displayed separately as contrasts; they cannot be attached to both mean bars.
- Holm adjustment applies to reported p-values, not these marginal or paired bootstrap CIs.
- CSFCube uses whole-query clusters and equal-facet macro averaging. Leave-one-out
  deltas are influence diagnostics, not 95% CIs. No LOO-derived error bars were used.
- All new marginal intervals are visualization summaries of frozen observations:
  20,000 bootstrap replicates, seed 20260929. Existing paired tests are unchanged.
- R−W p=0.968 is auxiliary/uncorrected; failure to reject is not proof of equivalence.
- Selection has zoomed y axes, explicitly marked; external and presentation bars start at zero.
- CSFCube exact means: SPECTER 0.511231/0.534238; heuristic 0.410580/0.414787
  (NDCG@10/AP). Exact saved contrast rounding is −0.101 [−0.157, −0.047]
  and −0.119 [−0.175, −0.065]; these supersede approximate spec values.

## Reproduction and provenance

Run `python3 plot_evidence.py` from this paper directory. No network or model calls.
`figure_statistics.json` records source SHA-256 hashes, marginal intervals and sampling details.
Frozen experiment JSON files are read-only inputs and are not overwritten.
''')
print('Generated four figures, manuscript variants, and statistical provenance.')
