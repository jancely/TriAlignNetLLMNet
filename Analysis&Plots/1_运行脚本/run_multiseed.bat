@echo off
REM Multi-seed runs for the full TriAlignNet model (seeds 1110 / 2222 / 3333)
cd /d %~dp0..

for %%P in (6 12 18 24) do (
    for %%S in (1110 2222 3333) do (
        echo === Seed %%S, pred_len=%%P ===
        python run.py --data Agricultural --model TriAlignNet --root_path ./Dataset/Dataset --data_path protein_output.csv --seq_len 24 --pred_len %%P --seed %%S --is_training 1 --des seed_%%S
    )
)
