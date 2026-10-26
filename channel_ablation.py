"""Run the channel experiment and remove an unused legacy timing placeholder."""
import argparse
import json
from pathlib import Path
from channel_ablation_core import (CHANNEL_SETS,STAT_NAMES,channel_stats,
    feature_columns,feature_names,check_normaliser,read_rows,base)
from channel_ablation_core import main as run_experiment

def clean_reports(out):
    for name in ['validation_predictions.csv','out_of_fold_predictions.csv']:
        path=out/name;rows=read_rows(path)
        if rows and 'decode_ms'in rows[0]:
            if any(float(r['decode_ms'])!=0 for r in rows):raise ValueError('Unexpected timing measurement')
            for r in rows:del r['decode_ms']
            base.write_csv(path,rows,list(rows[0]))
    p=out/'run_protocol.json';protocol=json.loads(p.read_text())
    protocol['timing_measurement']='Not collected in the ablation experiment; unused zero placeholder removed from reports.'
    protocol['entry_point_sha256']=base.sha(__file__)
    base.dump(p,protocol)
    render_plot(out)

def render_plot(out):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=read_rows(out/'channel_comparison.csv')
    methods=['logistic_regression','rbf_svm','random_forest'];variants=list(CHANNEL_SETS)
    fig,axes=plt.subplots(1,2,figsize=(13,5.5));x=np.arange(4);width=.25
    for ax,scope,title in [(axes[0],'grouped_training_cv','Six-session grouped CV'),(axes[1],'MAIN07_validation','MAIN07 validation')]:
        for j,name in enumerate(methods):
            rs={r['channel_set']:r for r in rows if r['scope']==scope and r['model']==name}
            ax.bar(x+(j-1)*width,[100*float(rs[v]['character_error_rate'])for v in variants],width,label=name.replace('_',' '))
        ax.set_xticks(x,['Existing\nenvelope','Z + waveform\nfeatures','XYZ accel','XYZ accel\n+ gyro'])
        ax.set_ylabel('Character error rate, including spaces (%)');ax.set_title(title)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
        ax.set_ylim(0,ax.get_ylim()[1]*1.08)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.suptitle('Channel-feature ablation: detector and classification algorithms held fixed')
    fig.tight_layout(rect=[0,.08,1,.94]);fig.savefig(out/'channel_comparison.png',dpi=160);plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--output',type=Path,default=Path('MorseChannels_Local'))
    args,_=p.parse_known_args()
    run_experiment();clean_reports(args.output)
