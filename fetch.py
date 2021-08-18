import wandb 
import sys

run = wandb.init('bdjobs')
artifact = run.use_artifact(sys.argv[1])
artifact_dir = artifact.download()