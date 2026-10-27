from pathlib import Path
import csv, collections, string
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parents[1]
ASSET=OUT/'MorseReportAssets'
(ASSET/'data').mkdir(exist_ok=True)
(ASSET/'figures').mkdir(exist_ok=True)
def align(a,b):
 d=[[0]*(len(b)+1) for _ in range(len(a)+1)]
 for i in range(len(a)+1):d[i][0]=i
 for j in range(len(b)+1):d[0][j]=j
 for i,x in enumerate(a,1):
  for j,y in enumerate(b,1):d[i][j]=min(d[i-1][j-1]+(x!=y),d[i-1][j]+1,d[i][j-1]+1)
 i,j=len(a),len(b);o=[]
 while i or j:
  if i and j and a[i-1]==b[j-1] and d[i][j]==d[i-1][j-1]:o.append((a[i-1],b[j-1],'match'));i-=1;j-=1
  elif i and j and d[i][j]==d[i-1][j-1]+1:o.append((a[i-1],b[j-1],'substitution'));i-=1;j-=1
  elif i and d[i][j]==d[i-1][j]+1:o.append((a[i-1],'<DEL>','deletion'));i-=1
  else:o.append(('<INS>',b[j-1],'insertion'));j-=1
 return list(reversed(o))

pred=list(csv.DictReader(open(OUT/'MorseFinal_Local/test_predictions.csv')))
base=list(string.ascii_uppercase+string.digits)+[' ']
rowlabels=base+['<INS>'];collabels=base+['?','<DEL>']
cm={};supports={};correct={};alignrows=[];stats=[]
methods=['rbf_svm','rbf_svm_enriched_z']
for method in methods:
 c=collections.Counter();s=collections.Counter();g=collections.Counter();edits=0
 rows=[r for r in pred if r['method']==method]
 assert len(rows)==144
 for r in rows:
  pairs=align(r['target_text'],r['predicted_text'])
  assert sum(k!='match' for x,y,k in pairs)==int(r['character_edits'])
  edits+=int(r['character_edits'])
  for n,(x,y,k) in enumerate(pairs,1):
   c[x,y]+=1
   if x!='<INS>':s[x]+=1;g[x]+=(x==y)
   alignrows.append([method,r['capture_id'],n,x,y,k])
 assert sum(s.values())==410
 assert edits==(37 if method=='rbf_svm' else 47)
 cm[method]=c;supports[method]=s;correct[method]=g
 for x in base:stats.append([method,'<SPACE>' if x==' ' else x,s[x],g[x],sum(c[x,y] for y in collabels if y!=x),c[x,'<DEL>'],100*g[x]/s[x]])
 # Keep all alphanumeric labels, spaces, invalid tokens and edit outcomes visible.
 counts=np.array([[c[x,y] for y in collabels] for x in rowlabels])
 den=np.maximum(counts.sum(axis=1),1)
 norm=counts/den[:,None]*100
 fig,ax=plt.subplots(figsize=(11.5,8.1))
 im=ax.imshow(norm,cmap='Blues',vmin=0,vmax=100,aspect='equal')
 display=lambda x:{' ':'SP','<INS>':'INS','<DEL>':'DEL'}.get(x,x)
 ax.set_xticks(range(len(collabels)),[display(x) for x in collabels],fontsize=9)
 ax.set_yticks(range(len(rowlabels)),[display(x) for x in rowlabels],fontsize=9)
 ax.tick_params(top=True,labeltop=True,bottom=False,labelbottom=False,length=0)
 ax.set_xlabel('Predicted character / deletion (DEL)',fontsize=10,labelpad=9)
 ax.xaxis.set_label_position('top')
 ax.set_ylabel('Intended character / insertion (INS)',fontsize=10)
 ax.set_xticks(np.arange(-.5,len(collabels),1),minor=True)
 ax.set_yticks(np.arange(-.5,len(rowlabels),1),minor=True)
 ax.grid(which='minor',color='#dfe4eb',linewidth=.35);ax.tick_params(which='minor',length=0)
 for i,j in zip(*np.nonzero(counts)):
  ax.text(j,i,str(counts[i,j]),ha='center',va='center',fontsize=8.6,color='white' if norm[i,j]>55 else '#172b4d')
 for spine in ax.spines.values():spine.set_visible(False)
 title='Original RBF SVM comparator' if method=='rbf_svm' else 'Enriched Z SVM primary candidate'
 fig.suptitle(title+' — final-test character confusion',fontsize=13,y=.99)
 cb=fig.colorbar(im,ax=ax,fraction=.025,pad=.025);cb.set_label('Percentage within row',fontsize=9);cb.ax.tick_params(labelsize=8)
 fig.text(.5,.025,'Cells: counts; blank cells: zero. SP: space. INS: extra predicted character. DEL: missing character.\nColour: percentage within each row; insertion row is normalised separately. Deterministic minimum-edit alignment.',ha='center',fontsize=8)
 fig.subplots_adjust(left=.07,right=.92,top=.89,bottom=.10)
 fig.savefig(ASSET/'figures'/('character_confusion_'+method+'.png'),dpi=300,bbox_inches='tight',pad_inches=.08)
 plt.close(fig)
 with open(ASSET/'data'/('character_confusion_'+method+'.csv'),'w') as f:
  w=csv.writer(f);w.writerow(['intended_or_insertion']+['<SPACE>' if x==' ' else x for x in collabels])
  for x in rowlabels:w.writerow(['<SPACE>' if x==' ' else x]+[c[x,y] for y in collabels])
with open(ASSET/'data/character_alignment.csv','w') as f:
 w=csv.writer(f);w.writerow(['method','capture_id','alignment_position','intended_character','predicted_character','operation']);w.writerows(alignrows)
with open(ASSET/'data/character_performance.csv','w') as f:
 w=csv.writer(f);w.writerow(['method','character','reference_support','correct','incorrect_or_deleted','deleted','aligned_recovery_percent']);w.writerows(stats)
assert sum(cm['rbf_svm'].values())==416
assert sum(cm['rbf_svm_enriched_z'].values())==418
print('Verified alignment: 410 reference characters per model, 37/47 edits, 6/8 insertions.')
