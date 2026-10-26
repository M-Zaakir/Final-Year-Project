"""Freeze development choices before a single reserved-test evaluation."""
from pathlib import Path
import argparse, json, sys, platform
import numpy as np
import pandas as pd
import joblib, sklearn
from scipy.signal import butter, sosfiltfilt
import baseline_decoder as base
import ml_compare as ml
import morse_pipeline as prep
import channel_ablation_core as channels

METHODS=['duration_baseline']+ml.MODEL_NAMES+['rbf_svm_enriched_z']
TEST_SESSIONS=['P01_MAIN08','P01_MAIN09']

def selected(manifest,split):
    return [r for r in manifest if r['include_primary']=='1' and r['split']==split]

def artifacts(prepared,baseline,models,channels_dir):
    paths={'manifest':prepared/'inclusion_manifest.csv','preprocessing':prepared/'preprocessing_config.json',
           'baseline':baseline/'baseline_model.json','channel_protocol':channels_dir/'run_protocol.json'}
    paths.update({n:models/'models'/(n+'.joblib') for n in ml.MODEL_NAMES})
    paths['rbf_svm_enriched_z']=channels_dir/'models'/'rbf_svm_az.joblib'
    paths.update({'code_'+n:Path(m.__file__) for n,m in [('baseline',base),('ml',ml),('preprocessing',prep),('channel',channels)]})
    paths['code_final']=Path(__file__)
    return paths

def freeze(prepared,baseline,models,channels_dir,out):
    if (out/'frozen_protocol.json').exists():raise ValueError('Protocol already frozen; use evaluate with this folder.')
    manifest=base.read_csv(prepared/'inclusion_manifest.csv');train=selected(manifest,'train');test=selected(manifest,'test')
    if len(train)!=423 or len(test)!=144:raise ValueError('Unexpected frozen cohort size')
    if {r['group_session'] for r in test}!=set(TEST_SESSIONS):raise ValueError('Unexpected test sessions')
    norm=json.loads((baseline/'baseline_model.json').read_text())
    if set(norm['training_capture_ids'])!={r['capture_id'] for r in train}:raise ValueError('Baseline training scope changed')
    if norm['manifest_sha256']!=base.sha(prepared/'inclusion_manifest.csv') or norm['preprocessing_sha256']!=base.sha(prepared/'preprocessing_config.json'):raise ValueError('Baseline source mismatch')
    cp=json.loads((channels_dir/'run_protocol.json').read_text())
    if cp['python_code_sha256']!=base.sha(channels.__file__) or cp['baseline_sha256']!=base.sha(baseline/'baseline_model.json'):raise ValueError('Channel source mismatch')
    for name in ml.MODEL_NAMES:
        b=ml.load_bundle(name,prepared,baseline,models)
        if b['fit_split']!='train' or set(b['training_capture_ids'])!={r['capture_id'] for r in train}:raise ValueError('ML training scope changed')
    z=joblib.load(channels_dir/'models'/'rbf_svm_az.joblib')
    if z['sklearn_version']!=sklearn.__version__:raise ValueError('Reproduce channel experiment with current scikit-learn first')
    if set(z['fit_capture_ids'])!={r['capture_id'] for r in train}:raise ValueError('Enriched model training scope changed')
    if z['channel_set']!='az' or z['model']!='rbf_svm' or z['normaliser']!=norm:raise ValueError('Enriched model configuration differs')
    if z['pulse_columns']!=channels.feature_columns('pulse','az') or z['gap_columns']!=channels.feature_columns('gap','az'):raise ValueError('Enriched feature schema differs')
    protocol=dict(version=1,methods=METHODS,primary_candidate='rbf_svm_enriched_z',
        selection_basis='Lowest pooled six-session training CV CER in the channel experiment; MAIN07 development evidence also considered. No test-based selection.',
        training_sessions=['P01_MAIN'+str(i).zfill(2) for i in range(1,7)],test_sessions=TEST_SESSIONS,
        fit_policy='Reuse train-only fitted models; no retraining on MAIN07 and no test-driven changes.',
        primary_metric='Pooled Levenshtein edits / total intended reference characters, including spaces; unknown tokens remain question marks.',
        secondary_metric='Exact intended-message recovery; no dictionary correction.',
        inclusion_policy='Existing include_primary=1 manifest rows only; no new performance-driven exclusions.',
        labels='Intended/count-aligned; not independently symbol-verified. Text metrics measure intended-target recovery.',
        generalisation_limits='One operator, same-day sessions, fixed mounting and recurring vocabulary; session split does not establish unseen-user/day/vocabulary generalisation.',
        preprocessing='Unchanged offline/noncausal 833 Hz interpolation grid, 120–230 Hz band, initial 0.25 s baseline, 20 ms RMS; acquisition was approximately 824 Hz.',
        runtime_claim='No embedded latency claim; no runtime measured here.',
        test_capture_ids=[r['capture_id'] for r in test],train_capture_ids=[r['capture_id'] for r in train],
        artifact_hashes={k:base.sha(p) for k,p in artifacts(prepared,baseline,models,channels_dir).items()},
        environment=dict(python=platform.python_version(),sklearn=sklearn.__version__,numpy=np.__version__),
        freeze_precedes_test_signal_loading=True)
    base.dump(out/'frozen_protocol.json',protocol)
    print('Frozen: 5 fixed methods; 144 reserved captures. No test signals read.',flush=True)

def infer(sensor,config,norm,bundles,z):
    """Prediction accepts only sensor/time data and frozen models, no labels."""
    expected=['timestamp_us']+prep.SENSOR_COLUMNS
    if list(sensor.columns)!=expected:raise ValueError('Inference requires sensor/time columns only')
    grid,full,_,_=prep.transform(sensor,config,all_channels=True);env=full[:,-1].astype(float);fs=float(config['resample_hz'])
    common=base.predict(env,fs,norm);predictions={'duration_baseline':common}
    for name in ml.MODEL_NAMES:predictions[name]=ml.predict_ml(env,fs,norm,bundles[name])
    band=sosfiltfilt(butter(4,[120,230],btype='bandpass',fs=fs,output='sos'),full[:,:6].astype(float),axis=0).astype(np.float32)
    f=channels.attach_features(dict(env=env,fs=fs,band=band,noise=np.std(band[grid<.25],axis=0)),norm)
    symbols=z['pulse'].predict(f['px'][:,z['pulse_columns']]).tolist() if len(f['px']) else []
    gaps=z['gap'].predict(f['gx'][:,z['gap_columns']]).tolist() if len(f['gx']) else []
    predictions['rbf_svm_enriched_z']=dict(text=ml.decode_symbols(symbols,gaps),symbols=symbols,gaps=gaps,intervals=common['intervals'])
    return predictions

def score(row,pred,reference):
    target=row['target_text'];text=pred['text'];pairs=base.match_events(pred['intervals'],reference)
    return dict(capture_id=row['capture_id'],group_session=row['group_session'],target_text=target,predicted_text=text,
        exact_message=int(target==text),character_edits=base.edit_distance(target,text),reference_characters=len(target),
        event_tp=len(pairs),event_fp=len(pred['intervals'])-len(pairs),event_fn=len(reference)-len(pairs),
        speed=row['speed'],strength=row['strength'],noise=row['noise'],condition_id=row['condition_id'],pair_id=row['pair_id'],
        complete_noise_pair=row['complete_noise_pair'])

def summary(rows):
    totals=base.summarize([dict(r,decode_ms=0) for r in rows]);totals.pop('median_decode_ms');return totals

def evaluate(dataset,prepared,baseline,models,channels_dir,out):
    protocol_path=out/'frozen_protocol.json';protocol=json.loads(protocol_path.read_text())
    # A started run may be incomplete: never silently repeat or overwrite it.
    if (out/'evaluation_started.json').exists():raise ValueError('Evaluation already started. Inspect saved outputs; do not overwrite or retune.')
    for key,path in artifacts(prepared,baseline,models,channels_dir).items():
        if base.sha(path)!=protocol['artifact_hashes'][key]:raise ValueError('Frozen artifact changed: '+key)
    manifest=base.read_csv(prepared/'inclusion_manifest.csv');rows=selected(manifest,'test')
    if [r['capture_id'] for r in rows]!=protocol['test_capture_ids']:raise ValueError('Test selection changed')
    if set(protocol['train_capture_ids'])&set(protocol['test_capture_ids']):raise ValueError('Train/test overlap')
    norm=json.loads((baseline/'baseline_model.json').read_text());config=json.loads((prepared/'preprocessing_config.json').read_text())
    bundles={n:ml.load_bundle(n,prepared,baseline,models) for n in ml.MODEL_NAMES}
    z=joblib.load(channels_dir/'models'/'rbf_svm_az.joblib')
    base.dump(out/'evaluation_started.json',dict(protocol_sha256=base.sha(protocol_path),test_records=len(rows)))
    results=[];events=[]
    for i,row in enumerate(rows,1):
        for field,hashfield in [('csv_relative','csv_sha256'),('json_relative','json_sha256')]:
            if base.sha(dataset/row[field])!=row[hashfield]:raise ValueError('Test source changed: '+row['capture_id'])
        raw=pd.read_csv(dataset/row['csv_relative'])
        predictions=infer(raw[['timestamp_us']+prep.SENSOR_COLUMNS],config,norm,bundles,z)
        # Ground truth is opened for scoring only after all method predictions.
        t0=int(raw.timestamp_us.iloc[0]);reference=np.array([[(a-t0)/1e6,(b-t0)/1e6] for state,a,b in prep.reference_runs(raw) if state==1]).reshape(-1,2)
        for method,pred in predictions.items():
            results.append(dict(method=method,**score(row,pred,reference)))
            for j,(a,b) in enumerate(pred['intervals']):events.append(dict(method=method,capture_id=row['capture_id'],pulse_index=j+1,start_s=float(a),end_s=float(b),symbol=pred['symbols'][j],following_gap=pred['gaps'][j] if j<len(pred['gaps']) else 'end'))
        if i%24==0:print(f'Test inference and scoring: {i}/{len(rows)}',flush=True)
    base.write_csv(out/'test_predictions.csv',results,list(results[0]));base.write_csv(out/'test_events.csv',events,list(events[0]))
    summaries=[dict(method=m,**summary([r for r in results if r['method']==m])) for m in METHODS]
    base.write_csv(out/'test_comparison.csv',summaries,list(summaries[0]))
    breakdown=[]
    for m in METHODS:
        mr=[r for r in results if r['method']==m]
        for field in ['group_session','speed','strength','noise','condition_id']:
            for value in sorted({r[field] for r in mr}):breakdown.append(dict(method=m,group=field,value=value,**summary([r for r in mr if r[field]==value])))
    base.write_csv(out/'test_breakdown.csv',breakdown,list(breakdown[0]))
    pairs=[]
    for m in METHODS:
        mr=[r for r in results if r['method']==m and r['complete_noise_pair']=='1']
        for pid in sorted({r['pair_id'] for r in mr}):
            members=[r for r in mr if r['pair_id']==pid]
            if len(members)!=2 or {r['noise'] for r in members}!={'clean','table_tap'}:raise ValueError('Invalid paired-noise group')
            c=next(r for r in members if r['noise']=='clean');n=next(r for r in members if r['noise']=='table_tap')
            pairs.append(dict(method=m,pair_id=pid,target_text=c['target_text'],clean_exact=c['exact_message'],noisy_exact=n['exact_message'],clean_edits=c['character_edits'],noisy_edits=n['character_edits'],reference_characters=c['reference_characters']))
    if pairs:base.write_csv(out/'test_paired_noise.csv',pairs,list(pairs[0]))
    render(out,summaries)
    base.dump(out/'complete.json',dict(protocol_sha256=base.sha(protocol_path),test_recordings=len(rows),methods=METHODS,
        files_sha256={p.name:base.sha(p) for p in sorted(out.glob('test_*')) if p.is_file()},final_test_consumed=True))
    print(json.dumps(summaries,indent=2))

def render(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['Duration','Logistic regression','RBF SVM','Random forest','Enriched Z SVM']
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    for ax,key,title in [(axes[0],'character_error_rate','Character error (%) — lower is better'),(axes[1],'exact_message_accuracy','Exact-message recovery (%) — higher is better')]:
        ax.barh(labels,[100*r[key] for r in rows],color=['#64748b']*4+['#168b82']);ax.invert_yaxis();ax.set_title(title,fontsize=10)
        for i,r in enumerate(rows):ax.text(100*r[key]+.3,i,f'{100*r[key]:.2f}',va='center',fontsize=9)
        ax.set_xlim(0,max(100*r[key] for r in rows)*1.19);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.suptitle('Reserved MAIN08–09: fixed development choices');fig.tight_layout();fig.savefig(out/'test_comparison.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('MorseDataset'));p.add_argument('--prepared',type=Path,default=Path('MorsePrepared'))
    p.add_argument('--baseline',type=Path,default=Path('MorseBaseline'));p.add_argument('--models',type=Path,default=Path('MorseML'))
    p.add_argument('--channels',type=Path,default=None);p.add_argument('--output',type=Path,default=Path('MorseFinal_Local'))
    p.add_argument('command',choices=['freeze','evaluate']);a=p.parse_args()
    cd=a.channels or (Path('MorseChannels_Local') if Path('MorseChannels_Local/complete.json').exists() or Path('MorseChannels_Local/run_protocol.json').exists() else Path('MorseChannels'))
    if a.command=='freeze':freeze(a.prepared,a.baseline,a.models,cd,a.output)
    else:evaluate(a.dataset,a.prepared,a.baseline,a.models,cd,a.output)

if __name__=='__main__':main()
