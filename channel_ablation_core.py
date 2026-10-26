"""Channel-feature ablation with fixed Z-based detection; no test signals loaded."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import csv
import hashlib
import joblib
import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter,sosfiltfilt
import sklearn
import baseline_decoder as base
import ml_compare as ml
from morse_pipeline import transform,SENSOR_COLUMNS

CHANNEL_SETS={'envelope_only':[], 'az':[2], 'accel_xyz':[0,1,2],
              'accel_gyro':[0,1,2,3,4,5]}
STAT_NAMES=['log_band_rms_snr','log_band_peak_snr','log_band_meanabs_snr',
            'band_crest_factor','band_abs_iqr_snr']
GROUPS=['P01_MAIN'+str(i).zfill(2)for i in range(1,7)]

def read_rows(path):
    with Path(path).open(newline='',encoding='utf-8')as f:return list(csv.DictReader(f))

def channel_stats(band,noise,intervals,fs):
    """Only sensor waveform/quiet statistics; no reference labels accepted."""
    band=np.asarray(band,float);noise=np.maximum(np.asarray(noise,float),1.)
    rows=[]
    for start,end in np.asarray(intervals).reshape(-1,2):
        lo=max(0,min(len(band)-1,int(np.floor(start*fs))))
        hi=max(lo+1,min(len(band),int(np.ceil(end*fs))))
        w=band[lo:hi];a=np.abs(w);rms=np.sqrt(np.mean(w*w,axis=0));peak=a.max(axis=0)
        iqr=np.quantile(a,.75,axis=0)-np.quantile(a,.25,axis=0)
        # Channel-major order, five features per channel.
        features=np.column_stack([np.log1p(rms/noise),np.log1p(peak/noise),
            np.log1p(a.mean(axis=0)/noise),peak/np.maximum(rms,1e-6),iqr/noise])
        rows.append(features.reshape(-1))
    x=np.asarray(rows,float).reshape(-1,30)
    if not np.isfinite(x).all():raise ValueError('Invalid channel features')
    return x

def feature_names(kind):
    existing=ml.PULSE_FEATURES if kind=='pulse'else ml.GAP_FEATURES
    return existing+[channel+'__'+stat for channel in SENSOR_COLUMNS for stat in STAT_NAMES]

def feature_columns(kind,variant):
    n=len(ml.PULSE_FEATURES if kind=='pulse'else ml.GAP_FEATURES)
    return list(range(n))+[n+5*c+j for c in CHANNEL_SETS[variant]for j in range(5)]

def load_records(dataset,prepared):
    manifest=read_rows(prepared/'inclusion_manifest.csv')
    config=json.loads((prepared/'preprocessing_config.json').read_text())
    selected=[r for r in manifest if r['include_primary']=='1'and r['split']in ['train','validation']]
    indexes={}
    for split in ['train','validation']:
        indexes.update({r['capture_id']:r for r in read_rows(prepared/('preprocessing_qc_'+split+'.csv'))})
    records=[];sos=butter(4,[120,230],btype='bandpass',fs=833.,output='sos')
    for i,row in enumerate(selected,1):
        if row['group_session']in ['P01_MAIN08','P01_MAIN09']:raise AssertionError('Test record selected')
        source=dataset/row['csv_relative']
        if base.sha(source)!=row['csv_sha256']:raise ValueError('Raw capture changed: '+row['capture_id'])
        if base.sha(dataset/row['json_relative'])!=row['json_sha256']:raise ValueError('Metadata changed')
        a=pd.read_csv(source)
        grid,full,_,_=transform(a,config,all_channels=True)
        index=indexes[row['capture_id']]
        with np.load(prepared/index['model_input_path'],allow_pickle=False)as z:
            names=z['feature_names'].tolist();env=z['X'][:,names.index('az_rms_envelope_raw')].astype(float);fs=float(z['sample_rate_hz'])
        if fs!=833.:raise ValueError('Expected fixed preprocessing grid')
        np.testing.assert_allclose(full[:,-1],env,atol=.05,rtol=1e-5)
        band=sosfiltfilt(sos,full[:,:6].astype(float),axis=0).astype(np.float32)
        quiet=np.std(band[grid<.25],axis=0)
        reference=read_rows(prepared/index['reference_path'])
        pulses=np.asarray([[float(r['start_s']),float(r['end_s'])]for r in reference if r['key_reference']=='1'])
        labels=[r['intended_symbol']for r in reference if r['key_reference']=='1']
        records.append(dict(row=row,env=env,fs=fs,band=band,noise=quiet,ref=pulses,labels=labels))
        if i%100==0:print(f'Prepared six-channel signals: {i}/{len(selected)}',flush=True)
    return records

def attach_features(record,normaliser):
    common=base.predict(record['env'],record['fs'],normaliser)
    intervals=common['intervals'];unit=common['unit_s']or np.exp(normaliser['timing']['log_unit_mean'])
    px,gx=ml.event_features(record['env'],record['fs'],intervals,unit)
    gi=np.column_stack([intervals[:-1,1],intervals[1:,0]])if len(intervals)>1 else np.empty((0,2))
    pc=channel_stats(record['band'],record['noise'],intervals,record['fs'])
    gc=channel_stats(record['band'],record['noise'],gi,record['fs'])
    return dict(px=np.column_stack([px,pc]),gx=np.column_stack([gx,gc]),common=common)

def training_arrays(records,features):
    xp=[];yp=[];xg=[];yg=[]
    for r,f in zip(records,features):
        pairs=base.match_events(f['common']['intervals'],r['ref']);mapping=dict(pairs)
        gaps=base.gap_labels(r['row']['target_morse'])
        for di,ri in pairs:xp.append(f['px'][di]);yp.append(r['labels'][ri])
        for di in range(len(f['gx'])):
            if di in mapping and di+1 in mapping and mapping[di+1]==mapping[di]+1:
                xg.append(f['gx'][di]);yg.append(gaps[mapping[di]])
    return np.asarray(xp),np.asarray(yp),np.asarray(xg),np.asarray(yg)

def fit_heads(arrays,name,variant):
    xp,yp,xg,yg=arrays;pc=feature_columns('pulse',variant);gc=feature_columns('gap',variant)
    return dict(pulse=ml.new_estimator(name).fit(xp[:,pc],yp),
                gap=ml.new_estimator(name).fit(xg[:,gc],yg),pulse_columns=pc,gap_columns=gc)

def evaluate(records,features,heads,name,variant,scope):
    sizes_p=[len(f['px'])for f in features];sizes_g=[len(f['gx'])for f in features]
    allp=np.vstack([f['px']for f in features]);allg=np.vstack([f['gx']for f in features])
    predicted_p=heads['pulse'].predict(allp[:,heads['pulse_columns']])
    predicted_g=heads['gap'].predict(allg[:,heads['gap_columns']])if len(allg)else np.array([])
    po=go=0;rows=[]
    for r,f,npulse,ngap in zip(records,features,sizes_p,sizes_g):
        symbols=predicted_p[po:po+npulse];gaps=predicted_g[go:go+ngap];po+=npulse;go+=ngap
        text=ml.decode_symbols(symbols,gaps);target=r['row']['target_text']
        pairs=base.match_events(f['common']['intervals'],r['ref'])
        rows.append(dict(scope=scope,model=name,channel_set=variant,capture_id=r['row']['capture_id'],
            group_session=r['row']['group_session'],target_text=target,predicted_text=text,
            exact_message=int(text==target),character_edits=base.edit_distance(target,text),
            reference_characters=len(target),event_tp=len(pairs),event_fp=npulse-len(pairs),
            event_fn=len(r['ref'])-len(pairs),decode_ms=0.,
            speed=r['row']['speed'],strength=r['row']['strength'],noise=r['row']['noise']))
    return rows

def summarize(rows):
    summaries=[]
    keys=sorted({(r['scope'],r['model'],r['channel_set'])for r in rows})
    for scope,name,variant in keys:
        rs=[r for r in rows if(r['scope'],r['model'],r['channel_set'])==(scope,name,variant)]
        stats=base.summarize(rs);stats.pop('median_decode_ms',None)
        summaries.append(dict(scope=scope,model=name,channel_set=variant,**stats))
    return summaries

def check_normaliser(model,train,held):
    ids={r['row']['capture_id']for r in train};hids={r['row']['capture_id']for r in held}
    if set(model['training_capture_ids'])!=ids:raise AssertionError('Normaliser fit scope mismatch')
    if ids&hids:raise AssertionError('Training/held-out overlap')

def run(dataset,prepared,baseline,cv,out):
    out.mkdir(parents=True,exist_ok=True)
    normaliser=json.loads((baseline/'baseline_model.json').read_text())
    if normaliser['manifest_sha256']!=base.sha(prepared/'inclusion_manifest.csv'):raise ValueError('Manifest changed')
    if normaliser['preprocessing_sha256']!=base.sha(prepared/'preprocessing_config.json'):raise ValueError('Preprocessing changed')
    inputs=json.loads((cv/'run_inputs.json').read_text())
    if inputs['manifest_sha256']!=normaliser['manifest_sha256']:raise ValueError('CV uses another manifest')
    if inputs['preprocessing_sha256']!=normaliser['preprocessing_sha256']:raise ValueError('CV preprocessing differs')
    if inputs['baseline_code_sha256']!=base.sha(base.__file__)or inputs['ml_code_sha256']!=base.sha(ml.__file__):
        raise ValueError('Decoder/ML code differs from completed grouped CV')
    records=load_records(dataset,prepared)
    train=[r for r in records if r['row']['split']=='train'];validation=[r for r in records if r['row']['split']=='validation']
    check_normaliser(normaliser,train,validation)
    validation_rows=[];cv_rows=[];provenance=[];model_sizes=[]
    def experiment(fit_records,held_records,norm,scope,save=False):
        tf=[attach_features(r,norm)for r in fit_records];hf=[attach_features(r,norm)for r in held_records]
        arrays=training_arrays(fit_records,tf);rows=[]
        for name in ml.MODEL_NAMES:
            for variant in CHANNEL_SETS:
                heads=fit_heads(arrays,name,variant)
                rows.extend(evaluate(held_records,hf,heads,name,variant,scope))
                if save:
                    path=out/'models'/(name+'_'+variant+'.joblib');path.parent.mkdir(exist_ok=True)
                    joblib.dump(dict(**heads,model=name,channel_set=variant,normaliser=norm,
                        feature_names_pulse=feature_names('pulse'),feature_names_gap=feature_names('gap'),
                        sklearn_version=sklearn.__version__,fit_capture_ids=[r['row']['capture_id']for r in fit_records]),path,compress=3)
                    model_sizes.append(dict(model=name,channel_set=variant,bundle_bytes=path.stat().st_size,
                        pulse_feature_count=len(heads['pulse_columns']),gap_feature_count=len(heads['gap_columns'])))
        return rows
    print('Full-training fit and MAIN07 channel comparison...',flush=True)
    validation_rows=experiment(train,validation,normaliser,'MAIN07_validation',save=True)
    base.write_csv(out/'validation_predictions.csv',validation_rows,list(validation_rows[0]))
    for heldout in GROUPS:
        fit_records=[r for r in train if r['row']['group_session']!=heldout]
        held_records=[r for r in train if r['row']['group_session']==heldout]
        norm=json.loads((cv/'folds'/heldout/'fold_baseline_model.json').read_text())
        check_normaliser(norm,fit_records,held_records)
        print('Channel CV: hold out '+heldout+'...',flush=True)
        cv_rows.extend(experiment(fit_records,held_records,norm,'grouped_training_cv'))
        provenance.append(dict(heldout_session=heldout,
            fit_ids=[r['row']['capture_id']for r in fit_records],heldout_ids=[r['row']['capture_id']for r in held_records],
            reused_verified_fold_normaliser=True,new_feature_scalers_and_classifiers_fit_on_training_fold_only=True))
    base.write_csv(out/'out_of_fold_predictions.csv',cv_rows,list(cv_rows[0]))
    summaries=summarize(validation_rows+cv_rows)
    base.write_csv(out/'channel_comparison.csv',summaries,list(summaries[0]))
    # Per-session CV results expose variation rather than just a pooled score.
    per_session=[]
    for group in GROUPS:
        for r in summarize([x for x in cv_rows if x['group_session']==group]):
            per_session.append(dict(heldout_session=group,**r))
    base.write_csv(out/'channel_cv_by_session.csv',per_session,list(per_session[0]))
    base.write_csv(out/'model_sizes.csv',model_sizes,list(model_sizes[0]))
    base.dump(out/'fold_provenance.json',provenance)
    base.dump(out/'run_protocol.json',dict(primary_recordings_train=len(train),validation=len(validation),
        channel_sets=CHANNEL_SETS,stats_per_channel=STAT_NAMES,
        preprocessing='Fixed 120--230 Hz band-pass per axis, centred raw counts; features normalised to first-0.25-s band noise (floor 1 raw count)',
        classifier_settings='Unchanged from first ML comparison; no hyperparameter search',
        test_loaded=False,raw_unchanged=True,detector='Identical Z detector and timing normaliser across channel sets within each fold',
        inference_channel_caveat='Measures additional classification features, not alternative detectors or raw-waveform models',
        python_code_sha256=base.sha(__file__),manifest_sha256=base.sha(prepared/'inclusion_manifest.csv'),
        baseline_sha256=base.sha(baseline/'baseline_model.json'),sklearn=sklearn.__version__,numpy=np.__version__,scipy=scipy.__version__,
        labels='Intended/count-aligned; not independently symbol-verified'))
    plot(out,summaries)
    print(json.dumps(summaries,indent=2),flush=True)

def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    variants=list(CHANNEL_SETS);x=np.arange(4);width=.25
    for ax,scope,title in [(axes[0],'grouped_training_cv','Six-session grouped CV'),(axes[1],'MAIN07_validation','MAIN07 validation')]:
        for j,name in enumerate(ml.MODEL_NAMES):
            rs={r['channel_set']:r for r in rows if r['scope']==scope and r['model']==name}
            ax.bar(x+(j-1)*width,[100*rs[v]['character_error_rate']for v in variants],width,label=name.replace('_',' '))
        ax.set_xticks(x,['Existing\nenvelope','Z + waveform\nfeatures','XYZ accel','XYZ accel\n+ gyro'])
        ax.set_ylabel('Character error rate, including spaces (%)');ax.set_title(title)
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8);fig.suptitle('Channel-feature ablation: detector and classification algorithms held fixed')
    fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'channel_comparison.png',dpi=160);plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,default=Path('MorseDataset'))
    p.add_argument('--prepared',type=Path,default=Path('MorsePrepared'))
    p.add_argument('--baseline',type=Path,default=Path('MorseBaseline'))
    p.add_argument('--cv',type=Path,default=None)
    p.add_argument('--output',type=Path,default=Path('MorseChannels_Local'))
    p.add_argument('command',choices=['run'])
    a=p.parse_args();cv=a.cv or(Path('MorseCV_Local')if Path('MorseCV_Local').exists()else Path('MorseCV'))
    run(a.dataset,a.prepared,a.baseline,cv,a.output)

if __name__=='__main__':main()
