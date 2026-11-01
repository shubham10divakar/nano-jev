# Runs the whole revision pipeline in order (training + evaluation + stats + tables).
# From code_repo\nano_jev:   powershell -ExecutionPolicy Bypass -File paper_revision\run_all.ps1
# Output of every step is appended to paper_revision\run_all.log.
# Steps whose output already exists are skipped, so you can stop and re-run it safely.
# Total: roughly 9-11 hours on an RTX 3060.

$env:PYTHONUTF8 = "1"
$log = "paper_revision\run_all.log"
$hasAggrefact = Test-Path "data_aggrefact\test.jsonl"
$base = "microsoft/MiniLM-L12-H384-uncased"

function Step($name, $done, [scriptblock]$cmd) {
    if ($done -and (Test-Path $done)) { Write-Host "skip  $name (exists: $done)"; return }
    $t = Get-Date
    Write-Host "start $name  $t"
    "`n===== $name  $t =====" | Out-File $log -Append -Encoding utf8
    & $cmd 2>&1 | Out-File $log -Append -Encoding utf8
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED $name (see $log)" -ForegroundColor Red }
    else { Write-Host ("done  $name  ({0:N0} min)" -f ((Get-Date) - $t).TotalMinutes) }
}

if (-not $hasAggrefact) { Write-Host "NOTE: data_aggrefact missing - AggreFact steps are skipped (see RUNBOOK step 0)" -ForegroundColor Yellow }
$testSets = @("data_clean", "data_heldout"); if ($hasAggrefact) { $testSets += "data_aggrefact" }

# ---------------- 1. training ----------------
foreach ($s in 0..4) {
    Step "train K-pass seed $s" "runs\v2\nanojev\seed$s\model.safetensors" { python scripts/train.py --data data_clean --base $base --epochs 3 --lr 5e-5 --seed $s --out runs/v2/nanojev/seed$s }
}
foreach ($s in 0..4) {
    Step "train 1-pass seed $s" "runs\v2\onepass\seed$s\heads.pt" { python scripts/onepass.py train --data data_clean --base $base --epochs 3 --lr 5e-5 --seed $s --out runs/v2/onepass/seed$s }
}
foreach ($f in 10, 25, 50) {
    Step "train ablation $f%" "runs\v2\ablation_data\frac$f\model.safetensors" { python scripts/train.py --data data_clean --base $base --epochs 3 --lr 5e-5 --seed 0 --train-fraction ($f / 100) --out runs/v2/ablation_data/frac$f }
}

# ---------------- 2. baselines (no training) ----------------
Step "baselines test"    "results\v2\baselines\llm.json"         { python scripts/baselines.py --which all --data data_clean --out results/v2/baselines }
Step "baselines heldout" "results\v2\baselines\llm_heldout.json" { python scripts/baselines.py --which all --data data_clean --test-data data_heldout --suffix _heldout --out results/v2/baselines }
if ($hasAggrefact) {
    Step "baselines aggrefact" "results\v2\baselines\llm_aggrefact.json" { python scripts/baselines.py --which all --data data_clean --test-data data_aggrefact --suffix _aggrefact --out results/v2/baselines }
}
Step "grounded NLI checkers" "results\v2\grounded\nli-MoritzLaurer_DeBERTa-v3-large-mnli-fever-anli-ling-wanli_data_heldout.json" { python scripts/baselines_grounded.py --systems nli:MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli nli:MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli --data data_clean --test-data $testSets --out results/v2/grounded }
Step "grounded MiniCheck" "results\v2\grounded\minicheck-deberta-v3-large_data_heldout.json" { python scripts/baselines_grounded.py --systems minicheck:roberta-large minicheck:deberta-v3-large --data data_clean --test-data $testSets --out results/v2/grounded }
if (Test-Path "ckpts\AlignScore-base.ckpt") {
    Step "grounded AlignScore" "results\v2\grounded\alignscore-ckpts_AlignScore-base_data_heldout.json" { python scripts/baselines_grounded.py --systems alignscore:ckpts/AlignScore-base.ckpt --data data_clean --test-data $testSets --out results/v2/grounded }
}
foreach ($t in "agnews", "yahoo", "emotion") {
    Step "custom Qwen $t" "results\v2\custom\llm_$t.json" { python scripts/baselines.py --which llm --data data_custom/$t --out results/v2/custom --suffix _$t }
}

# ---------------- 3. evaluation ----------------
foreach ($s in 0..4) {
    $m = "runs/v2/nanojev/seed$s"; $r = "results/v2/nanojev/seed$s"
    Step "eval K-pass seed $s test"    "$r\test.json"    { python scripts/evaluate.py --model $m --data data_clean --results-dir $r --name test }
    Step "eval K-pass seed $s heldout" "$r\heldout.json" { python scripts/evaluate.py --model $m --data data_clean --test-data data_heldout --results-dir $r --name heldout --no-save }
    if ($hasAggrefact) { Step "eval K-pass seed $s aggrefact" "$r\aggrefact.json" { python scripts/evaluate.py --model $m --data data_clean --test-data data_aggrefact --results-dir $r --name aggrefact --no-save } }
    $m = "runs/v2/onepass/seed$s"; $r = "results/v2/onepass/seed$s"
    Step "eval 1-pass seed $s test"    "$r\test.json"    { python scripts/onepass.py eval --model $m --data data_clean --results-dir $r --name test }
    Step "eval 1-pass seed $s heldout" "$r\heldout.json" { python scripts/onepass.py eval --model $m --data data_clean --test-data data_heldout --results-dir $r --name heldout }
    if ($hasAggrefact) { Step "eval 1-pass seed $s aggrefact" "$r\aggrefact.json" { python scripts/onepass.py eval --model $m --data data_clean --test-data data_aggrefact --results-dir $r --name aggrefact } }
}
if ($hasAggrefact) {
    Step "aggrefact re-fit demo" "results\v2\nanojev\seed0\aggrefact_refit.json" { python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_aggrefact --test-data data_aggrefact --results-dir results/v2/nanojev/seed0 --name aggrefact_refit --no-save }
}
foreach ($f in 10, 25, 50) {
    Step "eval ablation $f%" "results\v2\ablation_data\frac$f\test.json" { python scripts/evaluate.py --model runs/v2/ablation_data/frac$f --data data_clean --results-dir results/v2/ablation_data/frac$f --name test --no-save }
}
Step "grounded CaliDec chunk-max" "results\v2\grounded\nanojev-runs_v2_nanojev_seed0_data_heldout.json" { python scripts/baselines_grounded.py --systems nanojev:runs/v2/nanojev/seed0 --data data_clean --test-data $testSets --out results/v2/grounded }
Step "cascade all decisions" "results\v2\cascade\seed0\cascade.json" { python scripts/cascade.py --model runs/v2/nanojev/seed0 --data data_clean --test-data data_clean data_heldout --decisions relevance sufficient grounded --out results/v2/cascade/seed0 }
foreach ($t in "agnews", "yahoo", "emotion") {
    Step "custom revised $t" "results\v2\custom\v2\$t.json" { python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_custom/$t --results-dir results/v2/custom/v2 --name $t --no-save }
}
Step "custom probe" "results\v2\custom\probe_onepass.json" { python scripts/custom_probe.py --encoder runs/v2/onepass/seed0 --tasks data_custom/agnews data_custom/yahoo data_custom/emotion --out results/v2/custom/probe_onepass.json }
Step "truncation test"    "results\v2\truncation\test.json"    { python scripts/truncation_analysis.py --model runs/v2/nanojev/seed0 --data data_clean --preds results/v2/nanojev/seed0/test_preds.jsonl --out results/v2/truncation/test.json --write-balanced data_clean_balanced }
Step "truncation heldout" "results\v2\truncation\heldout.json" { python scripts/truncation_analysis.py --model runs/v2/nanojev/seed0 --data data_heldout --preds results/v2/nanojev/seed0/heldout_preds.jsonl --out results/v2/truncation/heldout.json --write-balanced data_heldout_balanced }
Step "eval balanced test"    "results\v2\nanojev\seed0\test_balanced.json"    { python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_clean --test-data data_clean_balanced --results-dir results/v2/nanojev/seed0 --name test_balanced --no-save }
Step "eval balanced heldout" "results\v2\nanojev\seed0\heldout_balanced.json" { python scripts/evaluate.py --model runs/v2/nanojev/seed0 --data data_clean --test-data data_heldout_balanced --results-dir results/v2/nanojev/seed0 --name heldout_balanced --no-save }
Step "latency GPU" "results\v2\latency\gpu.json" { python scripts/latency.py --systems nanojev:runs/v2/nanojev/seed0 onepass:runs/v2/onepass/seed0 reranker:BAAI/bge-reranker-v2-m3 llm:Qwen/Qwen3-4B-Instruct-2507 --device cuda --out results/v2/latency/gpu.json }
Step "latency CPU" "results\v2\latency\cpu.json" { python scripts/latency.py --systems nanojev:runs/v2/nanojev/seed0 onepass:runs/v2/onepass/seed0 reranker:BAAI/bge-reranker-v2-m3 llm:Qwen/Qwen3-4B-Instruct-2507 --device cpu --n 64 --out results/v2/latency/cpu.json }
Step "end-to-end RAG" "results\v2\e2e\e2e.json" { python scripts/e2e_rag.py --nanojev runs/v2/nanojev/seed0 --sets data_e2e/hotpot.jsonl data_e2e/musique.jsonl --out results/v2/e2e }

# ---------------- 4. statistics (always re-run, seconds) ----------------
Step "stats" $null {
    python scripts/stats.py seeds --name nanojev_test --preds "results/v2/nanojev/seed*/test_preds.jsonl"
    python scripts/stats.py seeds --name nanojev_heldout --preds "results/v2/nanojev/seed*/heldout_preds.jsonl"
    python scripts/stats.py seeds --name onepass_test --preds "results/v2/onepass/seed*/test_preds.jsonl"
    python scripts/stats.py compare --name nanojev_vs_onepass_test --a results/v2/nanojev/seed0/test_preds.jsonl --b results/v2/onepass/seed0/test_preds.jsonl
    python scripts/stats.py compare --name nanojev_vs_llm_test --a results/v2/nanojev/seed0/test_preds.jsonl --b results/v2/baselines/llm_preds.jsonl
    python scripts/stats.py compare --name nanojev_vs_llm_heldout --a results/v2/nanojev/seed0/heldout_preds.jsonl --b results/v2/baselines/llm_heldout_preds.jsonl
    python scripts/stats.py compare --name nanojev_vs_reranker_test --a results/v2/nanojev/seed0/test_preds.jsonl --b results/v2/baselines/reranker_preds.jsonl
    if ($hasAggrefact) { python scripts/stats.py compare --name nanojev_vs_minicheck_aggrefact --a results/v2/nanojev/seed0/aggrefact_preds.jsonl --b results/v2/grounded/minicheck-roberta-large_data_aggrefact_preds.jsonl }
    python scripts/stats.py cascade --name cascade_seed0 --cascade results/v2/cascade/seed0/cascade_preds.jsonl --cascade-json results/v2/cascade/seed0/cascade.json
}

# ---------------- 5. tables ----------------
python scripts/make_tables.py
Write-Host "`nAll steps attempted. Check the table summary above and any FAILED lines; full output in $log"
