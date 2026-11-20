import torch
import random
from omegaconf import OmegaConf
from hydra import initialize, compose
from pipeline import AlzheimerPipeline
from data.dataset import build_dataloaders
import warnings

# Suppress annoying warnings for the presentation
warnings.filterwarnings("ignore")

def main():
    print("\n" + "="*60)
    print("🧠 ADNI MULTI-AGENT AI SYSTEM — CLINICAL DEMONSTRATION 🧠")
    print("="*60 + "\n")
    
    # 1. Load configuration via Hydra
    with initialize(version_base=None, config_path="configs"):
        cfg = compose(config_name="config")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Get the test dataset (patients the AI has NEVER seen before)
    _, _, test_loader = build_dataloaders(cfg)
    dataset = test_loader.dataset
    
    # Pick a random patient
    patient_idx = random.randint(0, len(dataset) - 1)
    patient_data = dataset[patient_idx]
    
    # The dataloader returns unbatched items, we need to add a batch dimension
    batch = {k: (torch.nan_to_num(v.unsqueeze(0).to(device)) if isinstance(v, torch.Tensor) else v) 
             for k, v in patient_data.items()}
    
    print(f"🏥 Selected Random Test Patient: {batch['subject_id']}")
    print(f"📊 True Diagnosis (Hidden from AI): {cfg.class_names[batch['label'].item()]}")
    print("\n" + "-"*60)
    print("🤖 Chief Physician is analyzing the patient...")
    
    try:
        pipeline = AlzheimerPipeline(cfg, device=str(device))
        pipeline.load_checkpoint("checkpoints/pipeline_final.pt")
        pipeline.eval_mode()
        
        with torch.no_grad():
            final_result = pipeline.predict(batch)
            
        print("\n=== AI DIAGNOSIS REPORT ===")
        print(f"⚠️  Final Diagnosis: {final_result.diagnosis}")
        print(f"📈  Confidence:      {final_result.confidence * 100:.1f}%")
        
        print("\n=== TEAM RELIANCE ===")
        for modality, weight in final_result.modality_weights.items():
            print(f"    - {modality} Team Weight: {weight * 100:.1f}%")
            
        if final_result.disagreement_flag:
            print("\n⚠️ The sMRI and Features teams disagreed! Chief Physician had to arbitrate.")
            
        print("\n=== PROBABILITIES ===")
        for cls, prob in final_result.probabilities.items():
            print(f"    {cls}: {prob * 100:.1f}%")
            
    except Exception as e:
        print("\n❌ Error loading model. Make sure training has completely finished!")
        print(f"Details: {e}")

if __name__ == "__main__":
    main()
