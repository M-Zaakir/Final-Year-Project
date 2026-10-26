"""First ML comparison: shared detector, vibration-derived event features.

Targets/references are used only for supervised training labels and scoring.
predict_ml() takes only the envelope, sample rate, training-fitted baseline
and training-fitted classifiers. No condition or target metadata is accepted.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time
import platform
import sys

import joblib
import numpy as np
import sklearn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from baseline_decoder import (predict,load_split,match_events,gap_labels,MORSE,
                              edit_distance,summarize,write_csv,dump,sha)

PULSE_FEATURES=['log_duration_s','log_duration_over_unit','log_unit_s',
                'previous_pulse_over_unit','next_pulse_over_unit',
                'previous_gap_over_unit','next_gap_over_unit','has_previous','has_next',
                'median_envelope_over_active','peak_envelope_over_active',
                'envelope_coefficient_variation','rms_envelope_over_active']
GAP_FEATURES=['log_duration_s','log_duration_over_unit','log_unit_s',
              'previous_pulse_over_unit','next_pulse_over_unit',
              'mean_envelope_over_active','peak_envelope_over_active',
              'iqr_envelope_over_active']
MODEL_NAMES=['logistic_regression','rbf_svm','random_forest']

def read_model(prepared,baseline):
    m=json.loads((baseline/'baseline_model.json').read_text())
    if m['manifest_sha256']!=sha(prepared/'inclusion_manifest.csv'):raise ValueError('Manifest differs from fitted baseline')
    if m['preprocessing_sha256']!=sha(prepared/'preprocessing_config.json'):raise ValueError('Preprocessing differs from baseline')
    return m

def event_features(envelope,fs,intervals,unit):
    """Pure vibration features. No annotation arguments are allowed."""
    env=np.asarray(envelope,float);p=np.asarray(intervals,float).reshape(-1,2)
    if not np.isfinite(env).all() or not np.isfinite(p).all():raise ValueError('Nonfinite input')
    unit=max(.005,float(unit));active=max(float(np.quantile(env,.99)),1e-6)
    durations=np.maximum(.005,p[:,1]-p[:,0])
    gaps=np.maximum(.005,p[1:,0]-p[:-1,1])
    pulse_x=[];gap_x=[]
    def clip_ratio(x):return float(np.clip(x/unit,0,30))
    def window(start,end):
        lo=max(0,min(len(env)-1,int(np.floor(start*fs))))
        hi=max(lo+1,min(len(env),int(np.ceil(end*fs))))
        return env[lo:hi]
    for i,(start,end)in enumerate(p):
        w=window(start,end);mean=float(w.mean())
        pulse_x.append([np.log(durations[i]),np.log(durations[i]/unit),np.log(unit),
            clip_ratio(durations[i-1])if i else 0,
            clip_ratio(durations[i+1])if i+1<len(p)else 0,
            clip_ratio(gaps[i-1])if i else 0,
            clip_ratio(gaps[i])if i<len(gaps)else 0,
            int(i>0),int(i+1<len(p)),float(np.median(w))/active,float(w.max())/active,
            float(w.std())/max(mean,1e-6),float(np.sqrt(np.mean(w*w)))/active])
    for i,gap in enumerate(gaps):
        w=window(p[i,1],p[i+1,0]);q=np.quantile(w,[.25,.75])
        gap_x.append([np.log(gap),np.log(gap/unit),np.log(unit),
            clip_ratio(durations[i]),clip_ratio(durations[i+1]),
            float(w.mean())/active,float(w.max())/active,float(q[1]-q[0])/active])
    px=np.asarray(pulse_x,float).reshape(-1,len(PULSE_FEATURES))
    gx=np.asarray(gap_x,float).reshape(-1,len(GAP_FEATURES))
    if not np.isfinite(px).all()or not np.isfinite(gx).all():raise ValueError('Nonfinite features')
    return px,gx

def decode_symbols(symbols,gaps):
    if not len(symbols):return ''
    token=str(symbols[0]);word='';words=[]
    for i,gap in enumerate(gaps):
        if gap=='element':token+=str(symbols[i+1])
        else:
            word+=MORSE.get(token,'?');token=str(symbols[i+1])
            if gap=='word':words.append(word);word=''
    word+=MORSE.get(token,'?');words.append(word)
    return ' '.join(words)

def predict_ml(envelope,fs,baseline_model,classifiers):
    common=predict(envelope,fs,baseline_model)
    unit=common['unit_s'] if common['unit_s']is not None else np.exp(baseline_model['timing']['log_unit_mean'])
    px,gx=event_features(envelope,fs,common['intervals'],unit)
    symbols=classifiers['pulse'].predict(px).tolist() if len(px)else []
    gaps=classifiers['gap'].predict(gx).tolist() if len(gx)else []
    return dict(text=decode_symbols(symbols,gaps),symbols=symbols,gaps=gaps,
                intervals=common['intervals'],unit_s=common['unit_s'])

def new_estimator(name):
    if name=='logistic_regression':
        return make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=3000,class_weight='balanced',random_state=42))
    if name=='rbf_svm':
        return make_pipeline(StandardScaler(),SVC(C=1.,kernel='rbf',gamma='scale',class_weight='balanced'))
    if name=='random_forest':
        return RandomForestClassifier(n_estimators=200,max_depth=8,min_samples_leaf=3,
                    max_features='sqrt',class_weight='balanced_subsample',random_state=42,n_jobs=1)
    raise ValueError(name)

def fit(prepared,baseline,out):
    model=read_model(prepared,baseline);data=load_split(prepared,'train')
    xp=[];yp=[];xg=[];yg=[];details=[]
    for row,env,fs,ref,labels in data:
        common=predict(env,fs,model)
        unit=common['unit_s']or np.exp(model['timing']['log_unit_mean'])
        px,gx=event_features(env,fs,common['intervals'],unit)
        pairs=match_events(common['intervals'],ref);mapping={di:ri for di,ri in pairs}
        glabels=gap_labels(row['target_morse'])
        for di,ri in pairs:
            xp.append(px[di]);yp.append(labels[ri])
            details.append(dict(capture_id=row['capture_id'],session=row['group_session'],event_type='pulse',
                detected_index=di+1,reference_index=ri+1,label=labels[ri],
                **dict(zip(PULSE_FEATURES,px[di]))))
        for di in range(len(gx)):
            # A gap has a usable label only if its two detections match
            # consecutive reference pulses in the same recording.
            if di in mapping and di+1 in mapping and mapping[di+1]==mapping[di]+1:
                ri=mapping[di];xg.append(gx[di]);yg.append(glabels[ri])
                details.append(dict(capture_id=row['capture_id'],session=row['group_session'],event_type='gap',
                    detected_index=di+1,reference_index=ri+1,label=glabels[ri],
                    **dict(zip(GAP_FEATURES,gx[di]))))
    xp=np.asarray(xp);xg=np.asarray(xg);yp=np.asarray(yp);yg=np.asarray(yg)
    if set(yp)!={'.','-'}or set(yg)!={'element','character','word'}:raise ValueError('Missing training classes')
    feature_fields=['capture_id','session','event_type','detected_index','reference_index','label']+list(dict.fromkeys(PULSE_FEATURES+GAP_FEATURES))
    write_csv(out/'training_event_features.csv',details,feature_fields)
    fits=[]
    for name in MODEL_NAMES:
        start=time.perf_counter();pulse=new_estimator(name).fit(xp,yp);gap=new_estimator(name).fit(xg,yg)
        elapsed=time.perf_counter()-start
        bundle=dict(name=name,pulse=pulse,gap=gap,pulse_features=PULSE_FEATURES,gap_features=GAP_FEATURES,
                    baseline_sha256=sha(baseline/'baseline_model.json'),manifest_sha256=sha(prepared/'inclusion_manifest.csv'),
                    fit_split='train',training_capture_ids=[r[0]['capture_id']for r in data],
                    sklearn_version=sklearn.__version__,label_status='intended_count_aligned_not_independently_verified')
        target=out/'models'/(name+'.joblib');target.parent.mkdir(parents=True,exist_ok=True)
        joblib.dump(bundle,target,compress=3)
        fits.append(dict(model=name,training_recordings=len(data),training_pulses=len(yp),
                         training_gaps=len(yg),fit_seconds=elapsed,model_file_bytes=target.stat().st_size,
                         training_pulse_intended_label_agreement=float((pulse.predict(xp)==yp).mean()),
                         training_gap_intended_label_agreement=float((gap.predict(xg)==yg).mean())))
    write_csv(out/'fit_summary.csv',fits,list(fits[0]))
    dump(out/'training_provenance.json',dict(training_recordings=len(data),
        sklearn_version=sklearn.__version__,joblib_version=joblib.__version__,
        pulse_labels={str(k):int((yp==k).sum())for k in np.unique(yp)},
        gap_labels={str(k):int((yg==k).sum())for k in np.unique(yg)},
        selected_train_only=True,cross_validation='not performed; models prespecified and compared on MAIN07',
        baseline_model_sha256=sha(baseline/'baseline_model.json'),
        preprocessing_sha256=sha(prepared/'preprocessing_config.json'),
        manifest_sha256=sha(prepared/'inclusion_manifest.csv'),
        feature_columns_are_vibration_only=True))
    print(json.dumps(fits,indent=2))

def load_bundle(name,prepared,baseline,out):
    provenance=json.loads((out/'training_provenance.json').read_text())
    if provenance['sklearn_version']!=sklearn.__version__:
        raise ValueError('Saved classifiers use scikit-learn '+provenance['sklearn_version']+
                         '; run ml_compare.py fit in your current environment before evaluating.')
    bundle=joblib.load(out/'models'/(name+'.joblib'))
    if bundle['baseline_sha256']!=sha(baseline/'baseline_model.json'):raise ValueError('Baseline changed; refit ML comparison')
    if bundle['manifest_sha256']!=sha(prepared/'inclusion_manifest.csv'):raise ValueError('Manifest changed; refit ML comparison')
    if bundle['pulse_features']!=PULSE_FEATURES or bundle['gap_features']!=GAP_FEATURES:
        raise ValueError('Event feature schema changed; refit explicitly')
    return bundle

def evaluate(prepared,baseline,out,split):
    duration_model=read_model(prepared,baseline);data=load_split(prepared,split)
    summaries=[];breakdown=[];allpredictions=[];paired_results=[]
    # Rerun baseline on the same data/hardware for a consistent comparison.
    for name in ['duration_baseline']+MODEL_NAMES:
        bundle=load_bundle(name,prepared,baseline,out)if name!='duration_baseline'else None
        rows=[];events=[]
        for row,env,fs,ref,labels in data:
            start=time.perf_counter()
            pred=predict(env,fs,duration_model)if bundle is None else predict_ml(env,fs,duration_model,bundle)
            elapsed=(time.perf_counter()-start)*1000
            pairs=match_events(pred['intervals'],ref);mapping={di:ri for di,ri in pairs}
            intended=row['target_text'].strip().upper();text=pred['text']
            r=dict(model=name,capture_id=row['capture_id'],split=split,target_text=intended,predicted_text=text,
                exact_message=int(text==intended),character_edits=edit_distance(intended,text),reference_characters=len(intended),
                event_tp=len(pairs),event_fp=len(pred['intervals'])-len(pairs),event_fn=len(ref)-len(pairs),
                intended_symbol_agreements=sum(pred['symbols'][di]==labels[ri]for di,ri in pairs),
                matched_reference_pulses=len(pairs),speed=row['speed'],strength=row['strength'],noise=row['noise'],
                pair_id=row['pair_id'],complete_noise_pair=row['complete_noise_pair'],
                decode_ms=elapsed,interpretation='intended_text_recovery_not_verified_transcription_accuracy')
            rows.append(r);allpredictions.append(r)
            for di,(a,b)in enumerate(pred['intervals']):
                ri=mapping.get(di)
                events.append(dict(capture_id=row['capture_id'],detected_pulse=di+1,start_s=float(a),end_s=float(b),
                    predicted_symbol=pred['symbols'][di],following_gap=pred['gaps'][di]if di<len(pred['gaps'])else'end',
                    intended_reference_symbol=labels[ri]if ri is not None else'',
                    label_status='intended_count_aligned_not_independently_verified'))
        write_csv(out/(split+'_'+name+'_predictions.csv'),rows,list(rows[0]))
        write_csv(out/(split+'_'+name+'_events.csv'),events,list(events[0]))
        total=summarize(rows);total.update(model=name,split=split)
        total['intended_symbol_agreement']=sum(r['intended_symbol_agreements']for r in rows)/max(1,sum(r['matched_reference_pulses']for r in rows))
        summaries.append(total)
        pair_ids=sorted({r['pair_id']for r in rows if r['complete_noise_pair']=='1'})
        for pair in pair_ids:
            members=[r for r in rows if r['pair_id']==pair]
            if len(members)!=2 or{r['noise']for r in members}!={'clean','table_tap'}:
                raise ValueError('Invalid canonical noise pair: '+pair)
            clean=next(r for r in members if r['noise']=='clean')
            noisy=next(r for r in members if r['noise']=='table_tap')
            paired_results.append(dict(model=name,pair_id=pair,split=split,
                target_text=clean['target_text'],speed=clean['speed'],strength=clean['strength'],
                clean_capture=clean['capture_id'],noisy_capture=noisy['capture_id'],
                clean_exact=clean['exact_message'],noisy_exact=noisy['exact_message'],
                clean_cer=clean['character_edits']/clean['reference_characters'],
                noisy_cer=noisy['character_edits']/noisy['reference_characters']))
        for field in ['speed','strength','noise']:
            for value in sorted({r[field]for r in rows}):
                breakdown.append(dict(model=name,split=split,group=field,value=value,
                                      **summarize([r for r in rows if r[field]==value])))
    write_csv(out/(split+'_comparison.csv'),summaries,list(summaries[0]))
    write_csv(out/(split+'_breakdown.csv'),breakdown,list(breakdown[0]))
    if paired_results:write_csv(out/(split+'_paired_noise.csv'),paired_results,list(paired_results[0]))
    by_id={}
    for r in allpredictions:
        record=by_id.setdefault(r['capture_id'],dict(capture_id=r['capture_id'],target_text=r['target_text'],
            speed=r['speed'],strength=r['strength'],noise=r['noise']))
        record[r['model']+'_text']=r['predicted_text'];record[r['model']+'_exact']=r['exact_message']
    comparison=list(by_id.values())
    write_csv(out/(split+'_message_comparison.csv'),comparison,list(comparison[0]))
    if split=='validation':
        ranked=sorted(summaries,key=lambda r:(r['character_error_rate'],-r['exact_message_accuracy'],r['model']))
        dump(out/'validation_ranking.json',dict(selection_rule='Lowest validation CER, then highest exact-message recovery, then name',
            provisional_leader=ranked[0]['model'],results=ranked,
            caveat='One validation session; no final test, no cross-validation, labels are intended/count-aligned.'))
        make_figure(out,summaries)
    print(json.dumps(summaries,indent=2))

def make_figure(out,summaries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[r['model'].replace('_','\n')for r in summaries]
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,key,title in [(axes[0],'exact_message_accuracy','Exact intended-message recovery (%)'),
                          (axes[1],'character_error_rate','Character error rate, including spaces (%)')]:
        values=[r[key]*100 for r in summaries]
        bars=ax.bar(names,values,color=['#667585','#26749b','#488c71','#a26d37'])
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+.5,f'{v:.1f}',ha='center')
        ax.set_title(title,fontsize=10);ax.set_ylim(0,max(values)*1.18);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('MAIN07 validation: 71 recordings; shared detector and timing normaliser')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'validation_comparison.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('MorsePrepared'))
    p.add_argument('--baseline',type=Path,default=Path('MorseBaseline'))
    p.add_argument('--output',type=Path,default=Path('MorseML'))
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('fit')
    e=sub.add_parser('evaluate');e.add_argument('--split',choices=['train','validation','test'],default='validation')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='fit':fit(a.prepared,a.baseline,a.output)
    else:evaluate(a.prepared,a.baseline,a.output,a.split)

if __name__=='__main__':main()
