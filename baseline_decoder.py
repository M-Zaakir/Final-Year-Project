"""Offline, vibration-only duration baseline. Never pass labels to predict()."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.ndimage import binary_closing, binary_propagation
from scipy.optimize import linear_sum_assignment

LETTERS='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
CODES=['.-','-...','-.-.','-..','.','..-.','--.','....','..','.---','-.-','.-..',
       '--','-.','---','.--.','--.-','.-.','...','-','..-','...-','.--','-..-',
       '-.--','--..','-----','.----','..---','...--','....-','.....','-....',
       '--...','---..','----.']
MORSE=dict(zip(CODES,LETTERS))

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2),encoding='utf-8')
def read_csv(path):
    with Path(path).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))
def write_csv(path,rows,fields):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf-8')as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

def load_split(prepared,split):
    selected=[r for r in read_csv(prepared/'inclusion_manifest.csv')
              if r['include_primary']=='1'and r['split']==split]
    index={r['capture_id']:r for r in read_csv(prepared/('preprocessing_qc_'+split+'.csv'))}
    data=[]
    for row in selected:
        if row['capture_id']not in index:raise ValueError('Missing preprocessing index: '+row['capture_id'])
        entry=index[row['capture_id']]
        with np.load(prepared/entry['model_input_path'],allow_pickle=False)as z:
            names=z['feature_names'].tolist()
            env=z['X'][:,names.index('az_rms_envelope_raw')].astype(float)
            fs=float(z['sample_rate_hz'])
        ref=read_csv(prepared/entry['reference_path'])
        pulses=np.asarray([[float(r['start_s']),float(r['end_s'])]for r in ref if r['key_reference']=='1'])
        labels=[r['intended_symbol']for r in ref if r['key_reference']=='1']
        data.append((row,env,fs,pulses,labels))
    if not data:raise ValueError('No selected recordings: '+split)
    return data

def detect(envelope,fs,config):
    """Adaptive amplitude threshold and morphology; no labels/conditions used."""
    env=np.asarray(envelope,float)
    if env.ndim!=1 or len(env)<4 or not np.isfinite(env).all():raise ValueError('Invalid envelope')
    bg=env[:max(3,min(len(env),int(.25*fs)))]
    floor=float(np.median(bg));mad=1.4826*float(np.median(np.abs(bg-floor)))
    amplitude=max(0.,float(np.quantile(env,.99))-floor)
    on=max(floor+6*mad,floor+config['fraction']*amplitude,np.finfo(float).eps)
    off=max(floor+3*mad,floor+.60*(on-floor))
    mask=binary_propagation(env>=on,mask=env>=off)
    gap=max(1,round(config['close_gap_s']*fs))
    if gap>1:mask=binary_closing(mask,structure=np.ones(gap,dtype=bool))
    edges=np.diff(np.r_[False,mask,False].astype(np.int8))
    starts=np.flatnonzero(edges==1);ends=np.flatnonzero(edges==-1)
    intervals=np.asarray([(s/fs,e/fs)for s,e in zip(starts,ends)
                          if(e-s)/fs>=config['min_pulse_s']],float).reshape(-1,2)
    return intervals,dict(on=on,off=off,background_median=floor,background_mad=mad)

def match_events(detected,reference,tolerance=.15):
    """One-to-one onset matching within 150 ms, reported as a diagnostic."""
    if not len(detected)or not len(reference):return []
    cost=np.abs(detected[:,None,0]-reference[None,:,0])
    i,j=linear_sum_assignment(np.where(cost<=tolerance,cost,1e6))
    return [(int(a),int(b))for a,b in zip(i,j)if cost[a,b]<=tolerance]

def gap_labels(morse):
    labels=[]
    words=[w.strip().split()for w in morse.split('/')if w.strip()]
    for wi,word in enumerate(words):
        for ci,code in enumerate(word):
            for si,_ in enumerate(code):
                if si<len(code)-1:labels.append('element')
                elif ci<len(word)-1:labels.append('character')
                elif wi<len(words)-1:labels.append('word')
    return labels

def emission(log_durations,log_unit,means,sigmas,priors):
    residual=(log_durations[:,None,None]-log_unit[None,:,None]-means[None,None,:])/sigmas[None,None,:]
    return .5*residual**2+np.log(sigmas)[None,None,:]-np.log(priors)[None,None,:]

def classify_timing(intervals,model):
    if not len(intervals):return '',[],[],None
    pulses=np.maximum(intervals[:,1]-intervals[:,0],.005)
    gaps=np.maximum(intervals[1:,0]-intervals[:-1,1],.005)
    timing=model['timing'];mu=timing['log_unit_mean'];sigma=timing['log_unit_sigma']
    units=np.linspace(mu-2.5*sigma,mu+2.5*sigma,241)
    pc=timing['pulse'];gc=timing['gap']
    pl=emission(np.log(pulses),units,np.array(pc['means']),np.array(pc['sigmas']),np.array(pc['priors']))
    gl=emission(np.log(gaps),units,np.array(gc['means']),np.array(gc['sigmas']),np.array(gc['priors']))
    # Infer one shared timing scale from this message, with a training-derived
    # prior. No requested speed or target text is available here.
    score=pl.min(axis=2).sum(axis=0)+gl.min(axis=2).sum(axis=0)+.5*((units-mu)/sigma)**2
    best=int(np.argmin(score));symbols=['.'if x==0 else'-'for x in pl[:,best,:].argmin(axis=1)]
    breaks=[['element','character','word'][x]for x in gl[:,best,:].argmin(axis=1)]
    tokens=[];current=symbols[0]
    for i,gap in enumerate(breaks):
        if gap=='element':current+=symbols[i+1]
        else:
            tokens.append(current)
            if gap=='word':tokens.append('/')
            current=symbols[i+1]
    tokens.append(current)
    words=[];word=''
    for token in tokens:
        if token=='/':words.append(word);word=''
        else:word+=MORSE.get(token,'?')
    words.append(word)
    return ' '.join(words),symbols,breaks,float(np.exp(units[best]))

def predict(envelope,fs,model):
    """Predict from vibration envelope only. No row/reference is accepted."""
    detected,thresholds=detect(envelope,fs,model['detection'])
    corrected=detected.copy()
    if len(corrected):
        corrected[:,0]-=model['onset_bias_s'];corrected[:,1]-=model['release_bias_s']
        corrected[:,0]=np.maximum(corrected[:,0],0)
        corrected[:,1]=np.maximum(corrected[:,1],corrected[:,0]+.005)
    text,symbols,gaps,unit=classify_timing(corrected,model)
    return dict(text=text,intervals=corrected,symbols=symbols,gaps=gaps,unit_s=unit,thresholds=thresholds)

def fit(prepared,out):
    data=load_split(prepared,'train');scores=[]
    for fraction in [.12,.20,.30]:
        for close_gap in [.015,.030,.050]:
            cfg=dict(fraction=fraction,close_gap_s=close_gap,min_pulse_s=.045)
            tp=fp=fn=0
            for _,env,fs,ref,_ in data:
                detected,_=detect(env,fs,cfg);pairs=match_events(detected,ref)
                tp+=len(pairs);fp+=len(detected)-len(pairs);fn+=len(ref)-len(pairs)
            f1=2*tp/max(1,2*tp+fp+fn)
            scores.append(dict(**cfg,tp=tp,fp=fp,fn=fn,f1=f1))
    # Stable tie-break: less closing, then smaller threshold fraction.
    scores.sort(key=lambda r:(-r['f1'],r['close_gap_s'],r['fraction']))
    best=scores[0];cfg={k:best[k]for k in ['fraction','close_gap_s','min_pulse_s']}
    onset=[];release=[];units=[];pc={'.':[],'-':[]};gc={'element':[],'character':[],'word':[]}
    for row,env,fs,ref,labels in data:
        detected,_=detect(env,fs,cfg)
        for di,ri in match_events(detected,ref):
            onset.append(detected[di,0]-ref[ri,0]);release.append(detected[di,1]-ref[ri,1])
        if len(ref)!=len(labels)or not len(ref):continue
        durations=ref[:,1]-ref[:,0]
        unit=float(np.median(durations/np.array([1 if l=='.'else 3 for l in labels])))
        if unit<=0:continue
        units.append(unit)
        for duration,label in zip(durations,labels):pc[label].append(np.log(duration/unit))
        gaps=ref[1:,0]-ref[:-1,1];expected=gap_labels(row['target_morse'])
        if len(gaps)!=len(expected):raise ValueError('Reference gap-count mismatch')
        for duration,label in zip(gaps,expected):gc[label].append(np.log(max(.005,duration)/unit))
    def robust(values):
        med=float(np.median(values));mad=1.4826*float(np.median(np.abs(np.asarray(values)-med)))
        return med,max(.25,min(.65,mad))
    pulse=[robust(pc[l])for l in ['.','-']];gap=[robust(gc[l])for l in ['element','character','word']]
    log_units=np.log(units);unit_mu=float(np.median(log_units));unit_sigma=max(.35,float(np.std(log_units)))
    model=dict(version='duration_baseline_1.0',detection=cfg,
               onset_bias_s=float(np.median(onset)),release_bias_s=float(np.median(release)),
               timing=dict(log_unit_mean=unit_mu,log_unit_sigma=unit_sigma,
                           pulse=dict(means=[v[0]for v in pulse],sigmas=[v[1]for v in pulse],priors=[.5,.5]),
                           gap=dict(means=[v[0]for v in gap],sigmas=[v[1]for v in gap],priors=[.65,.30,.05])),
               fit_split='train',training_recordings=len(data),
               preprocessing_sha256=sha(prepared/'preprocessing_config.json'),
               manifest_sha256=sha(prepared/'inclusion_manifest.csv'),
               offline=True,label_status='intended_count_aligned_not_independently_verified',
               training_capture_ids=[r[0]['capture_id']for r in data])
    dump(out/'baseline_model.json',model)
    write_csv(out/'detection_training_search.csv',scores,list(scores[0]))
    print(json.dumps({k:v for k,v in model.items()if k not in ['training_capture_ids','timing']},indent=2))

def edit_distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]
        for j,y in enumerate(b,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=current
    return previous[-1]

def summarize(rows):
    n=len(rows);tp=sum(r['event_tp']for r in rows);fp=sum(r['event_fp']for r in rows);fn=sum(r['event_fn']for r in rows)
    return dict(recordings=n,exact_messages=sum(r['exact_message']for r in rows),
                exact_message_accuracy=sum(r['exact_message']for r in rows)/max(1,n),
                character_error_rate=sum(r['character_edits']for r in rows)/max(1,sum(r['reference_characters']for r in rows)),
                character_edits=sum(r['character_edits']for r in rows),
                reference_characters=sum(r['reference_characters']for r in rows),
                pulse_precision=tp/max(1,tp+fp),pulse_recall=tp/max(1,tp+fn),pulse_f1=2*tp/max(1,2*tp+fp+fn),
                event_tp=tp,event_fp=fp,event_fn=fn,
                median_decode_ms=float(np.median([r['decode_ms']for r in rows]))if n else None)

def evaluate(prepared,out,split):
    model=json.loads((out/'baseline_model.json').read_text())
    if model['preprocessing_sha256']!=sha(prepared/'preprocessing_config.json'):raise ValueError('Preprocessing config changed; refit explicitly')
    if model['manifest_sha256']!=sha(prepared/'inclusion_manifest.csv'):raise ValueError('Inclusion manifest changed; refit explicitly')
    data=load_split(prepared,split);results=[];reviews=[];examples=[];event_details=[]
    for row,env,fs,ref,labels in data:
        before=time.perf_counter();pred=predict(env,fs,model);elapsed=(time.perf_counter()-before)*1000
        pairs=match_events(pred['intervals'],ref)
        expected=row['target_text'].strip().upper();text=pred['text']
        symbol_agree=sum(pred['symbols'][di]==labels[ri]for di,ri in pairs)
        mapping={di:ri for di,ri in pairs}
        expected_gaps=gap_labels(row['target_morse'])
        gap_disagreements=(sum(a!=b for a,b in zip(pred['gaps'],expected_gaps))
                           if len(pred['gaps'])==len(expected_gaps)else '')
        for di,(start,end)in enumerate(pred['intervals']):
            ri=mapping.get(di)
            event_details.append(dict(capture_id=row['capture_id'],detected_pulse=di+1,
                start_s=float(start),end_s=float(end),duration_ms=float((end-start)*1000),
                predicted_symbol=pred['symbols'][di],
                following_gap=pred['gaps'][di]if di<len(pred['gaps'])else'end',
                matched_reference_pulse=ri+1 if ri is not None else '',
                intended_reference_symbol=labels[ri]if ri is not None else '',
                label_status='intended_count_aligned_not_independently_verified'))
        r=dict(capture_id=row['capture_id'],split=split,target_text=expected,predicted_text=text,
               exact_message=int(text==expected),character_edits=edit_distance(expected,text),
               reference_characters=len(expected),event_tp=len(pairs),event_fp=len(pred['intervals'])-len(pairs),
               event_fn=len(ref)-len(pairs),detected_pulses=len(pred['intervals']),reference_pulses=len(ref),
               intended_symbol_agreements=symbol_agree,matched_reference_pulses=len(pairs),
               intended_gap_disagreements=gap_disagreements,
               estimated_unit_ms=pred['unit_s']*1000 if pred['unit_s']is not None else '',
               speed=row['speed'],strength=row['strength'],noise=row['noise'],decode_ms=elapsed,
               interpretation='target_recovery_not_verified_transcription_accuracy')
        results.append(r)
        # Annotation flags are diagnostics only, never retrospective exclusions.
        # Fixed 2-unit pulse rule highlights possible intended/reference conflicts.
        if len(ref)==len(labels)and len(ref):
            durations=ref[:,1]-ref[:,0]
            u=np.median(durations/np.array([1 if x=='.'else 3 for x in labels]))
            for i,(duration,label)in enumerate(zip(durations,labels)):
                duration_label='.'if duration<2*u else'-'
                if label!=duration_label:
                    reviews.append(dict(capture_id=row['capture_id'],split=split,pulse_number=i+1,
                                        intended_symbol=label,duration_ms=duration*1000,record_unit_ms=u*1000,
                                        flag='duration_rule_disagrees_with_intended_symbol',action='review_not_auto_exclude'))
        if split=='validation'and len(examples)<6:examples.append((row,env,fs,pred,ref))
    write_csv(out/(split+'_predictions.csv'),results,list(results[0]))
    write_csv(out/(split+'_detected_events.csv'),event_details,
              ['capture_id','detected_pulse','start_s','end_s','duration_ms','predicted_symbol',
               'following_gap','matched_reference_pulse','intended_reference_symbol','label_status'])
    write_csv(out/(split+'_annotation_review.csv'),reviews,
              ['capture_id','split','pulse_number','intended_symbol','duration_ms','record_unit_ms','flag','action'])
    summary=summarize(results);summary['split']=split
    summary['important']='CER includes spaces and is measured against intended text, not an independently verified sent transcript. Reference-matching tolerance is 150 ms; no test evaluation occurred unless split=test explicitly.'
    dump(out/(split+'_summary.json'),summary)
    breakdown=[]
    for field in ['speed','strength','noise']:
        for value in sorted({r[field]for r in results}):breakdown.append(dict(group=field,value=value,**summarize([r for r in results if r[field]==value])))
    write_csv(out/(split+'_breakdown.csv'),breakdown,list(breakdown[0]))
    print(json.dumps(summary,indent=2))
    if examples:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(3,2,figsize=(14,9))
        for ax,(row,env,fs,pred,ref)in zip(axes.flat,examples):
            t=np.arange(len(env))/fs;ax.plot(t,env,lw=.8,color='#165c8a')
            for a,b in ref:ax.axvspan(a,b,color='#c88527',alpha=.12)
            for a,b in pred['intervals']:ax.hlines(env.max()*1.05,a,b,color='#a3324c',lw=2)
            ax.set_title(f"Intended: {row['target_text']} | Predicted: {pred['text']}",fontsize=10)
            ax.set_xlabel('Time (s)');ax.set_ylabel('Envelope (raw counts)');ax.grid(alpha=.15)
        fig.suptitle('Validation examples: envelope, button-reference shading, detected-pulse bars')
        fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(out/'validation_examples.png',dpi=140);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prepared',type=Path,default=Path('MorsePrepared'))
    p.add_argument('--output',type=Path,default=Path('MorseBaseline'))
    sub=p.add_subparsers(dest='command',required=True);sub.add_parser('fit')
    e=sub.add_parser('evaluate');e.add_argument('--split',choices=['train','validation','test'],default='validation')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.command=='fit':fit(a.prepared,a.output)
    else:evaluate(a.prepared,a.output,a.split)

if __name__=='__main__':main()
