import wandb
import sys
run = wandb.init(project="bdjobs",job_type='Graph_Update')
artifact = wandb.Artifact(sys.argv[1],type='dataset')
if sys.argv[1] == 'all':
    artifact.add_dir('./')
else:
    artifact.add_dir(sys.argv[1])
run.log_artifact(artifact)