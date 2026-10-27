"""Regenerate report figures from the included verified numeric snapshots."""
from pathlib import Path
import csv,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def read(name):
    with (ROOT/'data'/name).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))

def matrix(method,kind,labels):
    values={(r['intended'],r['predicted']):int(r['count'])for r in read('event_confusion.csv')if r['method']==method and r['event_type']==kind}
    return np.array([[values.get((a,b),0)for b in labels]for a in labels])

def draw(ax,counts,labels,title):
    pc=100*counts/counts.sum(axis=1,keepdims=True)
    ax.imshow(pc,vmin=0,vmax=100,cmap='Blues')
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j,i,f'{counts[i,j]}\n({pc[i,j]:.1f}%)',ha='center',va='center',fontsize=12,color='white'if pc[i,j]>55 else '#132b42')
    ax.set_xticks(range(len(labels)),labels);ax.set_yticks(range(len(labels)),labels)
    ax.set_xlabel('Predicted class');ax.set_ylabel('Intended class');ax.set_title(title,fontsize=12,pad=14)
    ax.tick_params(length=0)

def main():
    plt.rcParams.update({'font.size':11,'font.family':'DejaVu Sans'})
    out=ROOT/'figures';out.mkdir(exist_ok=True)
    a=matrix('rbf_svm','pulse',['.','-']);b=matrix('rbf_svm_enriched_z','pulse',['.','-'])
    assert a.sum()==1308 and np.array_equal(a,b)
    fig,ax=plt.subplots(figsize=(5,4.5));draw(ax,a,['Dot','Dash'],'Matched-pulse classification: both SVMs')
    fig.tight_layout();fig.savefig(out/'pulse_confusion.png',dpi=300,bbox_inches='tight');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.5))
    for ax,m,title in zip(axes,['rbf_svm','rbf_svm_enriched_z'],['Original SVM (comparator)','Enriched Z SVM (primary)']):
        a=matrix(m,'gap',['element','character','word']);assert a.sum()==1163
        draw(ax,a,['Element','Character','Word'],title)
    fig.tight_layout(w_pad=3);fig.savefig(out/'gap_confusion.png',dpi=300,bbox_inches='tight');plt.close(fig)
    rows=read('final_comparison.csv');labels=['Duration','Logistic\nregression','Original\nSVM','Random\nforest','Enriched Z\nSVM (primary)']
    fig,ax=plt.subplots(figsize=(9,4));bars=ax.bar(labels,[100*float(r['character_error_rate'])for r in rows],color=['#7b8da1']*4+['#21867a'])
    for bar,r in zip(bars,rows):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.25,f'{100*float(r["character_error_rate"]):.2f}%',ha='center',fontsize=10)
    ax.set_ylim(0,16);ax.set_ylabel('Character error rate (%)');ax.set_title('Reserved MAIN08–09: 144 recordings, 410 intended characters')
    ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(out/'final_cer.png',dpi=300);plt.close(fig)
    examples=json.loads((ROOT/'data'/'example_info.json').read_text());waves=read('example_waveforms.csv')
    fig,axes=plt.subplots(3,1,figsize=(12,9))
    for ax,info in zip(axes,examples):
        rs=[r for r in waves if r['capture_id']==info['capture_id']];t=np.array([float(r['time_s'])for r in rs]);e=np.array([float(r['envelope_raw'])for r in rs]);peak=e.max()
        ax.plot(t,e,lw=1.2,color='#166583',label='Z RMS envelope')
        for j,(a,b)in enumerate(info['reference_intervals']):ax.axvspan(a,b,color='#d99d38',alpha=.17,label='Button reference'if j==0 else None)
        for v in info['events']:ax.text((float(v['start_s'])+float(v['end_s']))/2,peak*1.06,v['symbol'],ha='center',fontsize=17)
        ax.set_ylim(0,peak*1.22);ax.set_xlabel('Time (s)',fontsize=17);ax.set_ylabel('Envelope\n(raw counts)',fontsize=16);ax.tick_params(labelsize=14);ax.grid(alpha=.12)
        ax.set_title(f"{info['title']}: intended {info['target']} / predicted {info['prediction']}\n{info['speed']}, {info['strength']} strength",fontsize=16)
    axes[0].legend(loc='upper right',fontsize=12)
    fig.tight_layout(h_pad=1.5);fig.savefig(out/'representative_signals.png',dpi=300,bbox_inches='tight');plt.close(fig)
    print('Four figures generated; matrix counts checked.')

if __name__=='__main__':main()
