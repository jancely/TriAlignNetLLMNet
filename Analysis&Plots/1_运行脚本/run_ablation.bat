@echo off
REM Ablation study: run each disabled component over pred_len 6/12/18/24
REM Usage: run from LLMNet-main/LLMNet-main directory
cd /d %~dp0..

for %%P in (6 12 18 24) do (
    echo === Ablation: disable MMD, pred_len=%%P ===
    python run.py --data Agricultural --model TriAlignNet --root_path ./Dataset/Dataset --data_path protein_output.csv --seq_len 24 --pred_len %%P --is_training 1 --des ablation_no_mmd --disable_mmd

    echo === Ablation: disable Anchor, pred_len=%%P ===
    python run.py --data Agricultural --model TriAlignNet --root_path ./Dataset/Dataset --data_path protein_output.csv --seq_len 24 --pred_len %%P --is_training 1 --des ablation_no_anchor --disable_anchor

    echo === Ablation: disable Fusion Transformer, pred_len=%%P ===
    python run.py --data Agricultural --model TriAlignNet --root_path ./Dataset/Dataset --data_path protein_output.csv --seq_len 24 --pred_len %%P --is_training 1 --des ablation_no_fusion --disable_fusion
)
