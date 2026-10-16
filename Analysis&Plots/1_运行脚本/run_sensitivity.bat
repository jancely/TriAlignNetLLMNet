@echo off
REM Sensitivity: input length (seq_len) and model dimension (d_model)
cd /d %~dp0..

for %%P in (6 12 18 24) do (
    for %%L in (36 48) do (
        echo === seq_len=%%L, pred_len=%%P ===
        python run.py --data Agricultural --model TriAlignNet --root_path ./Dataset/Dataset --data_path protein_output.csv --seq_len %%L --pred_len %%P --is_training 1 --des seqlen_%%L
    )
)

for %%P in (6 12 18 24) do (
    for %%D in (128 256) do (
        echo === d_model=%%D, pred_len=%%P ===
        python run.py --data Agricultural --model TriAlignNet --root_path ./Dataset/Dataset --data_path protein_output.csv --seq_len 24 --pred_len %%P --d_model %%D --d_ff %%D --is_training 1 --des dmodel_%%D
    )
)
