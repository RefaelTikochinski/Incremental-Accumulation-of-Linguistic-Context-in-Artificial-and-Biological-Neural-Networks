%%bash
#datalad install -r ///labs/hasson/narratives
cd ~/narratives
#download a specific subdataset use datalad get ~/dir

datalad get /home/refaelti/narratives/participants.tsv
#To download all relevant files of a specific task

task="milkyway"

datalad get /home/refaelti/narratives/sub-*/func/sub-*_task-$task\_events.tsv
datalad get /home/refaelti/narratives/derivatives/afni-smooth/tpl-fsaverage6/tpl-fsaverage6_hemi-*_desc-cortex_mask.gii
datalad get /home/refaelti/narratives/derivatives/afni-nosmooth/sub-*/func/sub-*_task-$task\_space-fsaverage6_hemi-*_desc-clean.func.gii
