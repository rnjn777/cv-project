import torch
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from hydra import initialize, compose
from pipeline import AlzheimerPipeline
from data.dataset import build_dataloaders
from train import train_assistant, train_branch_joint

def main():
    print("🚀 FAST FEATURE TRAINING PROTOCOL INITIATED")
    
    # 1. Fix the CSVs (replace NaNs with 0)
    print("Fixing missing data in CSVs...")
    for file in ["data/raw/clinical.csv", "data/raw/cognitive.csv"]:
        df = pd.read_csv(file)
        df.fillna(0, inplace=True)
        df.to_csv(file, index=False)
        
    # 2. Load config
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 3. Build dataloaders
    train_loader, val_loader, _ = build_dataloaders(cfg)
    
    # MONKEY PATCH: Disable MRI loading to make training 100x faster!
    # The clinical and cognitive models don't use the MRI tensor anyway.
    train_loader.dataset._load_nifti = lambda sid: torch.zeros(1, *train_loader.dataset.mri_shape, dtype=torch.float32)
    val_loader.dataset._load_nifti = lambda sid: torch.zeros(1, *val_loader.dataset.mri_shape, dtype=torch.float32)
    
    pipeline = AlzheimerPipeline(cfg, device=str(device))
    
    # 4. Train Clinical
    print("\n" + "="*50)
    print("TRAINING CLINICAL AGENT (High Speed)")
    print("="*50)
    train_assistant(
        pipeline.features_supervisor.clinical_assistant,
        input_key="clinical",
        train_loader=train_loader,
        val_loader=val_loader,
        cfg_training=cfg.training,
        device=device,
        checkpoint_dir="checkpoints",
        tag="clinical",
        num_classes=3
    )
    
    # 5. Train Cognitive
    print("\n" + "="*50)
    print("TRAINING COGNITIVE AGENT (High Speed)")
    print("="*50)
    train_assistant(
        pipeline.features_supervisor.cognitive_assistant,
        input_key="cognitive",
        train_loader=train_loader,
        val_loader=val_loader,
        cfg_training=cfg.training,
        device=device,
        checkpoint_dir="checkpoints",
        tag="cognitive",
        num_classes=3
    )
    
    print("\n✅ FAST TRAINING COMPLETE!")
    print("You now have all 4 models fully trained. Run `python rescue.py` to assemble them!")

if __name__ == "__main__":
    main()
