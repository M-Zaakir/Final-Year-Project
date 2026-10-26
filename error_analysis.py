"""Post-test diagnostics only: read saved predictions, never fit or retune."""
import argparse,csv,json,hashlib,collections
from pathlib import Path
import numpy as np
import pandas as pd
import baseline_decoder as base
import ml_compare as ml
import morse_pipeline as prep

FOCUS=['rbf_svm','rbf_svm_enriched_z']

def write(path,rows):
    if rows:base.write_csv(path,rows,list(rows[0]))

def text_edits(a,b):
    """One deterministic minimum-edit alignment; ties prefer match/sub, del, ins."""
    d=np.zeros((len(a)+1,len(b)+1),int);d[:,0]=np.arange(len(a)+1);d[0,:]=np.arange(len(b)+1)
    for i,x in enumerate(a,1):
        for j,y in enumerate(b,1):d[i,j]=min(d[i-1,j]+1,d[i,j-1]+1,d[i-1,j-1]+(x!=y))
    counts=collections.Counter();i,j=len(a),len(b)
    while i or j:
        if i and j and d[i,j]==d[i-1,j-1]+(a[i-1]!=b[j-1]):
            if a[i-1]!=b[j-1]:counts['substitutions']+=1
            i-=1;j-=1
        elif i and d[i,j]==d[i-1,j]+1:counts['deletions']+=1;i-=1
        else:counts['insertions']+=1;j-=1
    return {k:counts[k] for k in ['substitutions','deletions','insertions']}

def category(missed,extra,pulse,gap):
    kinds=[]
    if missed or extra:kinds.append('detection')
    if pulse:kinds.append('pulse_label')
    if gap:kinds.append('gap_label')
    return '+'.join(kinds) or 'none'

def validate(result):
    c=json.loads((result/'complete.json').read_text());p=json.loads((result/'frozen_protocol.json').read_text())
    if base.sha(result/'frozen_protocol.json')!=c['protocol_sha256']:raise ValueError('Protocol hash mismatch')
    for name,digest in c['files_sha256'].items():
        if base.sha(result/name)!=digest:raise ValueError('Output hash mismatch: '+name)
    if set(p['train_capture_ids'])&set(p['test_capture_ids']):raise ValueError('Fit/test overlap')
    return p

def analyse(dataset,prepared,result,out,reference=None):
    protocol=validate(result);out.mkdir(parents=True,exist_ok=True)
    predictions=base.read_csv(result/'test_predictions.csv');saved=base.read_csv(result/'test_events.csv')
    methods=protocol['methods'];ids=set(protocol['test_capture_ids']);manifest=base.read_csv(prepared/'inclusion_manifest.csv')
    selected={r['capture_id']:r for r in manifest if r['include_primary']=='1' and r['split']=='test'}
    if set(selected)!=ids:raise ValueError('Source manifest cohort differs')
    if len(predictions)!=len(ids)*len(methods):raise ValueError('Incomplete prediction table')
    byevent=collections.defaultdict(list)
    for r in saved:byevent[(r['method'],r['capture_id'])].append(r)
    cache={};cfg=json.loads((prepared/'preprocessing_config.json').read_text())
    for i,(cid,row) in enumerate(selected.items(),1):
        for field,hashfield in [('csv_relative','csv_sha256'),('json_relative','json_sha256')]:
            if base.sha(dataset/row[field])!=row[hashfield]:raise ValueError('Raw source mismatch: '+cid)
        raw=pd.read_csv(dataset/row['csv_relative']);t0=int(raw.timestamp_us.iloc[0])
        refs=np.array([[(s-t0)/1e6,(e-t0)/1e6] for k,s,e in prep.reference_runs(raw) if k==1]).reshape(-1,2)
        labels=[x for x in row['target_morse'] if x in '.-']
        if len(labels)!=len(refs):raise ValueError('Reference pulse count is not aligned')
        cache[cid]=dict(row=row,reference=refs,labels=labels,gap_labels=base.gap_labels(row['target_morse']))
        if i%48==0:print(f'Checked raw reference sources: {i}/{len(ids)}',flush=True)
    details=[];pulse_rows=[];gap_rows=[];confusion=collections.Counter();summaries=[]
    seen=set()
    for r in predictions:
        key=r['method'],r['capture_id']
        if key in seen:raise ValueError('Duplicate method/capture')
        seen.add(key);v=cache[r['capture_id']];row=v['row']
        for k in ['target_text','group_session','speed','strength','noise','condition_id','pair_id','complete_noise_pair']:
            if r[k]!=row[k]:raise ValueError('Prediction/source metadata differs: '+k)
        ev=sorted(byevent[key],key=lambda e:int(e['pulse_index']));intervals=np.array([[float(e['start_s']),float(e['end_s'])]for e in ev]).reshape(-1,2)
        if ml.decode_symbols([e['symbol']for e in ev],[e['following_gap']for e in ev[:-1]])!=r['predicted_text']:raise ValueError('Saved events do not reconstruct text')
        pairs=base.match_events(intervals,v['reference']);mapping=dict(pairs);pm=gm=0
        for di,ri in pairs:
            actual=ev[di]['symbol'];expected=v['labels'][ri];wrong=int(actual!=expected);pm+=wrong
            confusion[(r['method'],'pulse',expected,actual)]+=1
            pulse_rows.append(dict(method=r['method'],capture_id=r['capture_id'],detected_index=di+1,reference_index=ri+1,intended_symbol=expected,predicted_symbol=actual,mismatch=wrong,reference_duration_s=v['reference'][ri,1]-v['reference'][ri,0],detected_duration_s=intervals[di,1]-intervals[di,0]))
        comparable=0
        for di in range(max(0,len(ev)-1)):
            if di in mapping and di+1 in mapping and mapping[di+1]==mapping[di]+1:
                ri=mapping[di];expected=v['gap_labels'][ri];actual=ev[di]['following_gap'];wrong=int(expected!=actual);gm+=wrong;comparable+=1
                confusion[(r['method'],'gap',expected,actual)]+=1
                gap_rows.append(dict(method=r['method'],capture_id=r['capture_id'],detected_index=di+1,reference_index=ri+1,intended_gap=expected,predicted_gap=actual,mismatch=wrong,reference_duration_s=v['reference'][ri+1,0]-v['reference'][ri,1],detected_duration_s=intervals[di+1,0]-intervals[di,1]))
        missed=len(v['reference'])-len(pairs);extra=len(ev)-len(pairs)
        for name,value in [('event_tp',len(pairs)),('event_fp',extra),('event_fn',missed),('character_edits',base.edit_distance(r['target_text'],r['predicted_text'])),('exact_message',int(r['target_text']==r['predicted_text'])),('reference_characters',len(r['target_text']))]:
            if int(r[name])!=value:raise ValueError('Saved score differs: '+name)
        details.append(dict(**r,matched_pulses=len(pairs),missed_pulses=missed,extra_pulses=extra,pulse_label_mismatches=pm,comparable_gaps=comparable,gap_label_mismatches=gm,uncomparable_detected_gaps=max(0,len(ev)-1)-comparable,diagnostic_category=category(missed,extra,pm,gm),invalid_morse_tokens=r['predicted_text'].count('?'),**text_edits(r['target_text'],r['predicted_text'])))
    for m in methods:
        ds=[r for r in details if r['method']==m];failed=[r for r in ds if r['exact_message']=='0'];ps=[r for r in pulse_rows if r['method']==m];gs=[r for r in gap_rows if r['method']==m]
        summaries.append(dict(method=m,recordings=len(ds),failed_messages=len(failed),character_edits=sum(int(r['character_edits'])for r in ds),pulse_mismatches=sum(r['mismatch']for r in ps),matched_pulses=len(ps),gap_mismatches=sum(r['mismatch']for r in gs),comparable_gaps=len(gs),missed_pulses=sum(r['missed_pulses']for r in ds),extra_pulses=sum(r['extra_pulses']for r in ds),failed_with_pulse_mismatch=sum(r['pulse_label_mismatches']>0 for r in failed),failed_with_gap_mismatch=sum(r['gap_label_mismatches']>0 for r in failed),failed_with_detection_mismatch=sum(r['missed_pulses']+r['extra_pulses']>0 for r in failed),messages_with_invalid_tokens=sum(r['invalid_morse_tokens']>0 for r in ds),invalid_tokens=sum(r['invalid_morse_tokens']for r in ds)))
    write(out/'message_diagnostics.csv',details);write(out/'failed_messages.csv',[r for r in details if r['exact_message']=='0'])
    write(out/'pulse_diagnostics.csv',pulse_rows);write(out/'gap_diagnostics.csv',gap_rows);write(out/'error_summary.csv',summaries)
    write(out/'event_confusion.csv',[dict(method=k[0],event_type=k[1],intended=k[2],predicted=k[3],count=v)for k,v in sorted(confusion.items())])
    cats=[]
    for m in methods:
        counts=collections.Counter(r['diagnostic_category']for r in details if r['method']==m and r['exact_message']=='0')
        cats.extend(dict(method=m,category=k,failed_messages=v)for k,v in sorted(counts.items()))
    write(out/'failure_categories.csv',cats)
    comparisons=[]
    lookup={(r['method'],r['capture_id']):r for r in details}
    for cid in sorted(ids):
        a,b=[lookup[(m,cid)]for m in FOCUS];comparisons.append(dict(capture_id=cid,target_text=a['target_text'],strength=a['strength'],noise=a['noise'],original_text=a['predicted_text'],enriched_text=b['predicted_text'],original_edits=int(a['character_edits']),enriched_edits=int(b['character_edits']),edit_change_enriched_minus_original=int(b['character_edits'])-int(a['character_edits']),original_exact=int(a['exact_message']),enriched_exact=int(b['exact_message'])))
    write(out/'svm_feature_comparison.csv',comparisons)
    pair_results=[]
    for m in methods:
        pairs=[r for r in base.read_csv(result/'test_paired_noise.csv')if r['method']==m]
        pair_results.append(dict(method=m,pairs=len(pairs),reference_characters_per_condition=sum(int(r['reference_characters'])for r in pairs),clean_edits=sum(int(r['clean_edits'])for r in pairs),noisy_edits=sum(int(r['noisy_edits'])for r in pairs),clean_exact=sum(int(r['clean_exact'])for r in pairs),noisy_exact=sum(int(r['noisy_exact'])for r in pairs),both_exact=sum(r['clean_exact']=='1'and r['noisy_exact']=='1'for r in pairs),clean_only_exact=sum(r['clean_exact']=='1'and r['noisy_exact']=='0'for r in pairs),noisy_only_exact=sum(r['clean_exact']=='0'and r['noisy_exact']=='1'for r in pairs)))
    write(out/'paired_noise_summary.csv',pair_results)
    agreement={}
    if reference:
        for name,keycols in [('test_predictions.csv',['method','capture_id']),('test_comparison.csv',['method'])]:
            local=base.read_csv(result/name);ref=base.read_csv(reference/name);refmap={tuple(r[k]for k in keycols):r for r in ref}
            changed=[]
            for row in local:
                other=refmap[tuple(row[k]for k in keycols)];dif={k:dict(local=row[k],reference=other[k])for k in row if row[k]!=other[k]}
                if dif:changed.append(dict(identity={k:row[k]for k in keycols},differences=dif))
            agreement[name]=dict(rows=len(local),changed_rows=changed)
    base.dump(out/'verification.json',dict(local_result_hashes_valid=True,local_results_authoritative=True,source_test_cohort_ids_match=True,source_metadata_matches_predictions=True,raw_csv_json_hashes_verified_against_available_manifest=True,available_manifest_byte_hash_matches_local_protocol=base.sha(prepared/'inclusion_manifest.csv')==protocol['artifact_hashes']['manifest'],manifest_note='Local manifest itself not supplied; its recorded byte hash differs. Capture IDs, targets, conditions and raw-source checks use the available original manifest. No claim of byte-identical local raw files.',local_environment=protocol['environment'],reference_comparison=agreement))
    plots(out,details,pulse_rows,gap_rows,summaries,dataset,cache,byevent,cfg)
    base.dump(out/'analysis_provenance.json',dict(code_sha256=base.sha(__file__),input_protocol_sha256=base.sha(result/'frozen_protocol.json'),input_result_hashes={p.name:base.sha(p)for p in result.glob('*.csv')},scope='Post-test diagnostic analysis only. No refitting, retuning, relabelling or exclusions.',event_label_caveat='Mismatches are against intended/count-aligned references; they do not independently establish transmitted Morse or operator versus decoder fault.',gap_alignment='Only gaps between consecutive matched reference pulses are compared. Other detected gaps remain unscored.',error_categories='Detection/pulse/gap mismatch flags may co-occur. Disjoint combinations are saved separately.',edit_alignment='Minimum Levenshtein alignment; ties prefer diagonal then deletion then insertion. Edit-operation counts are not causal attribution.'))
    print(json.dumps(summaries,indent=2))

def plots(out,details,pulses,gaps,summaries,dataset,cache,events,cfg):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=['Duration','Logistic regression','RBF SVM','Random forest','Enriched Z SVM']
    fig,ax=plt.subplots(figsize=(10,5))
    x=np.arange(5);w=.25
    for j,(k,label)in enumerate([('failed_with_pulse_mismatch','Pulse label'),('failed_with_gap_mismatch','Gap label'),('failed_with_detection_mismatch','Detection')]):ax.bar(x+(j-1)*w,[r[k]for r in summaries],w,label=label)
    ax.set_xticks(x,names,rotation=12);ax.set_ylabel('Failed messages with mismatch');ax.set_title('Diagnostic flags against intended/count-aligned references\nFlags can co-occur; counts are not additive');ax.legend();ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);fig.tight_layout();fig.savefig(out/'failure_flags.png',dpi=180);plt.close(fig)
    # Examples fixed for interpretation, not for changing model settings.
    cases=[('P01_MAIN08_recording_20261015_141537','Pulse-label disagreement'),('P01_MAIN09_recording_20261015_150833','Gap-label disagreement'),('P01_MAIN09_recording_20261015_152431','Unmatched reference pulse')]
    fig,axes=plt.subplots(3,1,figsize=(13,9))
    examples=[]
    for ax,(cid,title)in zip(axes,cases):
        v=cache[cid];row=v['row'];raw=pd.read_csv(dataset/row['csv_relative']);grid,x,_,_=prep.transform(raw,cfg);env=x[:,-1];peak=max(float(env.max()),1)
        ax.plot(grid,env,color='#166583',lw=.9,label='Z RMS envelope')
        for j,(a,b)in enumerate(v['reference']):
            ax.axvspan(a,b,color='#d99d38',alpha=.16,label='Button interval (reference only)'if j==0 else None)
        ev=events[('rbf_svm',cid)];intervals=np.array([[float(e['start_s']),float(e['end_s'])]for e in ev]);pairs=base.match_events(intervals,v['reference']);matched={ri for _,ri in pairs}
        for e in ev:ax.text((float(e['start_s'])+float(e['end_s']))/2,peak*1.05,e['symbol'],ha='center',va='bottom',fontsize=10)
        for j,(a,b)in enumerate(v['reference']):
            if j not in matched:ax.annotate('Unmatched reference',xy=((a+b)/2,peak*.12),xytext=((a+b)/2,peak*.65),arrowprops=dict(arrowstyle='->',color='#c53a3a'),ha='center',fontsize=8)
        d=next(r for r in details if r['capture_id']==cid and r['method']=='rbf_svm')
        ax.set_title(f"{title}: intended {row['target_text']!r} → SVM {d['predicted_text']!r}; {row['speed']}, {row['strength']} strength",fontsize=10)
        ax.set_ylabel('Envelope (raw counts)');ax.set_xlabel('Time (s)');ax.set_ylim(0,peak*1.23);ax.grid(alpha=.12)
        examples.append(dict(capture_id=cid,description=title,target_text=row['target_text'],predicted_text=d['predicted_text'],diagnostic_category=d['diagnostic_category']))
    axes[0].legend(loc='upper right',fontsize=8);fig.suptitle('Representative final-test diagnostics; symbols above pulses are original SVM outputs\nShading gives button timing, not independently verified Morse labels',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'representative_signals.png',dpi=180);plt.close(fig);write(out/'representative_cases.csv',examples)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('MorseDataset'));p.add_argument('--prepared',type=Path,default=Path('MorsePrepared'))
    p.add_argument('--results',type=Path,default=Path('MorseFinal_Local'));p.add_argument('--reference',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path('MorseErrors_Local'));a=p.parse_args();analyse(a.dataset,a.prepared,a.results,a.output,a.reference)

if __name__=='__main__':main()
