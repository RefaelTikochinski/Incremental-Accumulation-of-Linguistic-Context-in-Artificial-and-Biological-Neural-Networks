
import extract_embeddings
import ridge
from sklearn.decomposition import PCA
from scipy.stats import pearsonr

import numpy as np
'''import pickle
with open('schaefer_atlas_400.pkl', 'rb') as f:
    atlas = pickle.load(f)
'''

def cos_sim(a,b):
    from numpy.linalg import norm
    return np.dot(a,b)/(norm(a)*norm(b))


def Voxel_based_encoder(vectors,transcript,neural_data ,warmup=False, shuffle=False,bootstrap=True ,folds=5, alpha=0.0):
    import ridge


    from scipy.stats import pearsonr


    Y=neural_data[:,[int(a) for a in set(transcript.TR_onset)]].transpose()
    X=np.array([vectors[transcript.TR_onset==TR,:].mean(axis=0) for TR in set(transcript.TR_onset)])

    if warmup:
        Y=Y[20:,:]
        X=X[20:,:]

    from sklearn.model_selection import KFold 

    kf=KFold(folds,shuffle=shuffle)
    kf.get_n_splits(X)

    results=[]
    
    if bootstrap:
        
        ind=int(X.shape[0]*0.6)
        chunklen=10
        nchunks=int(ind//5/chunklen)

        for train_ind, test_ind in kf.split(X):

            X_train=X[train_ind,:]
            X_test=X[test_ind,:]
            Y_train=Y[train_ind,:]
            Y_test=Y[test_ind,:]
            results_=ridge.bootstrap_ridge(X_train, Y_train, X_test, Y_test, alphas=np.logspace(0, 3, 20), nboots=20, chunklen=chunklen, nchunks=nchunks)
            results.append(results_[1])

        mean_results=np.array(results).mean(axis=0)
    
    else:
        for train_ind, test_ind in kf.split(X):

            X_train=X[train_ind,:]
            X_test=X[test_ind,:]
            Y_train=Y[train_ind,:]
            Y_test=Y[test_ind,:]
            W=ridge.ridge(X_train,Y_train,alpha=alpha)
            pred=np.dot(X_test,W)


            corrs=[]
            for i in range(Y.shape[1]):
                Pred_i=pred[:,i]
                Y_i=Y_test[:,i]
                corr=np.arctanh(pearsonr(Pred_i,Y_i)[0])
                corrs.append(corr)
            
            results.append(corrs)

        kf_results=np.array([r for r in results])
        mean_results=kf_results.mean(axis=0)

    return {"mean_results": mean_results }


def Voxel_based_encoder_r2(vectors,transcript,neural_data, alpha=0.0):
    import ridge


    from scipy.stats import pearsonr


    Y=neural_data[:,[int(a) for a in set(transcript.TR_onset)]].transpose()
    X=np.array([vectors[transcript.TR_onset==TR,:].mean(axis=0) for TR in set(transcript.TR_onset)])

    import ridge
    from sklearn.metrics import r2_score

    W=ridge.ridge(X,Y,alpha=alpha)
    pred=np.dot(X,W)

    R2=r2_score(Y,pred,multioutput="raw_values")

    return {"mean_results": R2 }

def pca_embed(v,n):
    p=PCA(n)
    return p.fit_transform(v)


def data_from_img(img,mask):
    inds=[]
    signals=[]
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            for k in range(mask.shape[2]):
                if mask[i,j,k]==1:
                    inds.append([i,j,k])
                    signals.append(img[i,j,k,:])

    neural_data=np.array(signals)
    return {"neural_data":neural_data,"coord":inds}

def data_to_img(data,coord, shape , affine=mask.affine):
    import nibabel
    img=np.zeros(shape=shape)
    for i in range(len(coord)):
        img[coord[i][0],coord[i][1],coord[i][2]]=data[i]
    img_=nibabel.Nifti1Image(img,affine=affine)

    return img_




#run per model per sub per task

for task in ["lucy","notthefallintact","sherlock","milkyway","tunnel","pieman","bronx","merlin"]:
    import get_data
    import numpy as np
    import pandas as pd
    from nilearn import surface, plotting, image, masking

    participants=pd.read_table("~/narratives/participants.tsv")


    if task=="pieman":
        pieman_spec={"exclude":["001","013", "014","021", "022", "038", "056","068","069"],"two_sessions":["002", "003", "004", "005", "006", "008", "010", "011", "012", "015","016"]}
        participants_in_task=[participants.participant_id[i] for i in range(len(participants)) if task in participants.task[i] and "piemanpni" not in participants.task[i]]
    else:
        participants_in_task=[participants.participant_id[i] for i in range(len(participants)) if task in participants.task[i]]

    if task=="milkyway":
        participants_in_task=[participants.participant_id[i] for i in range(len(participants)) if task in participants.task[i] and "original" in participants.condition[i]]

    subs=[a.split("-")[1] for a in participants_in_task]
    #subs=[s for s in subs if s!="004" and s!="013"]


    sub=subs[-1]

    onset=pd.read_table("~/narratives/sub-"+sub+"/func/sub-"+sub+"_task-"+task+"_events.tsv").onset[0]

    neural_data=get_data.get_data(sub,task)

    if task=="milkyway":
        transcript=pd.read_csv("~/narratives/stimuli/gentle/milkywayoriginal/align.csv",names=["word_orig","word","onset","offset"])
    else:
        transcript=pd.read_csv("~/narratives/stimuli/gentle/"+task+"/align.csv",names=["word_orig","word","onset","offset"])
    transcript=transcript.dropna().reset_index(drop=True)

    TR=1.5
    transcript["TR_onset"]=(onset+transcript.onset)//TR
    transcript["TR_offset"]=(onset+transcript.offset)//TR

    last_TR=int(transcript.TR_offset.max())

    import json 
    j  = open("/data/home/refaelti/narratives/code/scan_exclude.json", "r")  
    j_=j.read() 
    subs_exc=json.loads(j_)   
    subs=[s for s in subs if s not in [a.split("-")[1] for a in subs_exc[task].keys()]] 

    
    import pickle
    with open('/data/home/refaelti/promptBased/representations_chatGPT_ws8_'+task+'.pkl', 'rb') as f:
        vectors_ = pickle.load(f)


    results_all_models_per_subs=[]
    for sub in subs:
        
        print(sub)
        if task=="pieman" and sub in pieman_spec["two_sessions"]:
            task_="pieman_run-1"
        else:
            task_=task

        import nilearn
        img=nilearn.image.load_img("/data/home/refaelti/narratives/derivatives/afni-nosmooth/sub-"+sub+"/func/sub-"+sub+"_task-"+task_+"_space-MNI152NLin2009cAsym_res-native_desc-clean_bold.nii.gz")
        mask_isc=nilearn.image.load_img("/data/home/refaelti/mask_isc_all_tasks_MNI152.nii.gz")
        mask=nilearn.image.load_img("/data/home/refaelti/narratives/derivatives/afni-smooth/tpl-MNI152NLin2009cAsym/tpl-MNI152NLin2009cAsym_res-"+task+"_desc-brain_mask.nii.gz")

        coord=data_from_img(np.asanyarray(img.dataobj),np.asanyarray(mask.dataobj))["coord"]
        coord_maskedISC=data_from_img(np.asanyarray(img.dataobj),np.asanyarray(mask_isc.dataobj))["coord"]

        neural_data=data_from_img(np.asanyarray(img.dataobj),np.asanyarray(mask.dataobj))["neural_data"]
        neural_data_maskedISC=data_from_img(np.asanyarray(img.dataobj),np.asanyarray(mask_isc.dataobj))["neural_data"]
        neural_data=neural_data_maskedISC

    #train for different layers
        folds=5
        alpha=0.0
        transcript_=transcript
        results_all_models=[]
        models=[ 'ws_8',"bl_ws", 'ws_32', 'ws_64', 'ws_128', 'ws_256', 'full','increment_sum_ws_32']
        if task in ["tunnel","merlin" ,"sherlock"]:
            models=[ 'ws_8',"bl_ws", 'ws_32', 'ws_64', 'ws_128', 'ws_256','increment_sum_ws_32']
        #models= vectors_.keys()
        for k in models:
            X=vectors_[k].astype("float")
            X=pca_embed(vectors_[k],32)
            #results=Voxel_based_encoder_corrall(X, transcript_, neural_data, warmup=True, shuffle=False,bootstrap=False, folds=folds)  
            results=Voxel_based_encoder_r2(X, transcript_, neural_data)  

            results_all_models.append(results)

        results_all_models=np.array([results_all_models[i]["mean_results"] for i in range(len(results_all_models))])
        results_all_models_per_subs.append(results_all_models)

    results_all_models_per_subs=np.array(results_all_models_per_subs)

    import pickle
    with open('results_all_models_per_subs_R2_'+task+'.pkl', 'wb') as f:
        pickle.dump(results_all_models_per_subs, f)



# encoding data all subs all tasks
# encoding data all subs all tasks
participants=pd.read_table("~/narratives/participants.tsv")
subs_ids=dict()
for task in ["lucy","merlin","notthefallintact","sherlock","milkyway","tunnel","bronx","pieman"]:
    

    if task=="pieman":
        pieman_spec={"exclude":["001","013", "014","021", "022", "038", "056","068","069"],"two_sessions":["002", "003", "004", "005", "006", "008", "010", "011", "012", "015","016"]}
        participants_in_task=[participants.participant_id[i] for i in range(len(participants)) if task in participants.task[i] and "piemanpni" not in participants.task[i]]
    else:
        participants_in_task=[participants.participant_id[i] for i in range(len(participants)) if task in participants.task[i]]

    if task=="milkyway":
        participants_in_task=[participants.participant_id[i] for i in range(len(participants)) if task in participants.task[i] and "original" in participants.condition[i]]

    subs=[a.split("-")[1] for a in participants_in_task]
    sub=subs[-1]
    onset=pd.read_table("~/narratives/sub-"+sub+"/func/sub-"+sub+"_task-"+task+"_events.tsv").onset[0]

    #neural_data=get_data.get_data(sub,task)

    if task=="milkyway":
        transcript=pd.read_csv("~/narratives/stimuli/gentle/milkywayoriginal/align.csv",names=["word_orig","word","onset","offset"])
    else:
        transcript=pd.read_csv("~/narratives/stimuli/gentle/"+task+"/align.csv",names=["word_orig","word","onset","offset"])
    transcript=transcript.dropna().reset_index(drop=True)

    TR=1.5
    transcript["TR_onset"]=(onset+transcript.onset)//TR
    transcript["TR_offset"]=(onset+transcript.offset)//TR

    last_TR=int(transcript.TR_offset.max())

    import json 
    j  = open("/data/home/refaelti/narratives/code/scan_exclude.json", "r")  
    j_=j.read() 
    subs_exc=json.loads(j_)   
    subs=[s for s in subs if s not in [a.split("-")[1] for a in subs_exc[task].keys()]] 

    subs_ids[task]=subs



TRs={"lucy":351,"merlin":612,"pieman":286,"notthefallintact":383,"sherlock":723,"milkyway":282,"tunnel":1013,"bronx":374}

tasks=np.concatenate([[task]*len(subs_ids[task]) for task in subs_ids.keys()], axis=0)
ids=np.concatenate([subs_ids[task] for task in subs_ids.keys()], axis=0)
len(ids),len(tasks)

import pickle
results_all_models_all_subs=[]
for task in ["lucy","merlin","notthefallintact","sherlock","milkyway","tunnel","bronx","pieman"]:
    url="/data/home/refaelti/promptBased/results_all_models_per_subs_R2_"+task+".pkl"

    with open(url, 'rb') as f:
        results_all_models = pickle.load(f)
        
    if task in ["tunnel","merlin" ,"sherlock"]:
        results_all_models = np.stack([results_all_models[:,i,:] for i in [0,1,2,3,4,5,5,-1]],axis=1)
    

    def adj_R2(R2,n,k):
        return 1-((1-R2)*(n-1)/(n-k-1))

    #results_all_models=np.array(adj_R2(results_all_models, TRs[task],32))
    results_all_models_all_subs.append(results_all_models)

results_all_models_all_subs=np.concatenate(results_all_models_all_subs,axis=0)
results_all_models_all_subs=np.tanh(results_all_models_all_subs)


remove_outliers=True
if remove_outliers:
    good_ind=[i for i in range(results_all_models_all_subs.shape[0]) if np.nanmax(np.nanmax(results_all_models_all_subs,axis=-1),axis=-1)[i] > 0.4]
    #good_ind=[i for i in range(results_all_models_all_subs.shape[0]) if results_all_models_all_subs[:,7,:].max(axis=-1)[i] > 0.2]
    results_all_models_all_subs=results_all_models_all_subs[good_ind,:,:]

results_all_models_all_subs.shape
