Have both combined loss and arc face loss with partial fully connected layer v2, arch - r100, vit
ms1mv2_r100 - r100
wf42m_pfc03_40epoch_8gpu_vit_b - vit_b
command - python train_v2_fine_tune_combined_loss.py configs/wf42m_pfc03_40epoch_8gpu_vit_b
command - python train_v2_fine_tune_arcface.py configs/ms1mv3_r100
Not included any test for validation data during training
