#!/usr/bin/env bash
# Sequentially run E1 -> E4 -> E5 on AutoDL with timestamped logs.
# Designed to be launched via nohup so it survives ssh disconnect.
#
# Logs go to /root/autodl-tmp/VSR-Place/runs/<utc>/{e1,e4,e5}.log
# Status sentinel files: pipeline.{started,e1_done,e4_done,e5_done,finished,failed}

set -uo pipefail
export PATH="/root/miniconda3/bin:$PATH"
REPO=/root/autodl-tmp/VSR-Place
cd "$REPO"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR=$REPO/runs/$STAMP
mkdir -p "$RUN_DIR"

log() {
    echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$RUN_DIR/pipeline.log"
}

mark() { touch "$RUN_DIR/pipeline.$1"; }

mark started
log "pipeline started, run_dir=$RUN_DIR"
log "GPU info:"
nvidia-smi --query-gpu=name,memory.used,memory.free --format=csv,noheader 2>&1 | tee -a "$RUN_DIR/pipeline.log"

# ---- E1: main NeurIPS run -------------------------------------------------
log "=== E1: run_main_neurips.py ==="
python3 scripts/run_main_neurips.py 2>&1 | tee "$RUN_DIR/e1.log"
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
    log "E1 failed with status $status"
    mark failed
    exit 1
fi
mark e1_done
log "E1 done. Result rows:"
python3 -c "import json; d=json.load(open('results/ispd2005/main_neurips.json')); print(f'{len(d)} rows; ok rows: {sum(1 for r in d if r.get(\"baseline_v\") is not None)}')" 2>&1 | tee -a "$RUN_DIR/pipeline.log"

# ---- E4: timestep sweep ---------------------------------------------------
log "=== E4: run_timestep_sweep.py ==="
python3 scripts/run_timestep_sweep.py 2>&1 | tee "$RUN_DIR/e4.log"
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
    log "E4 failed with status $status"
    mark failed
    exit 1
fi
mark e4_done
log "E4 done."

# ---- E5: memory profile ---------------------------------------------------
log "=== E5: run_mem_profile.py ==="
python3 scripts/run_mem_profile.py 2>&1 | tee "$RUN_DIR/e5.log"
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
    log "E5 failed with status $status (non-fatal: bigblue OOMs are expected)"
fi
mark e5_done
log "E5 done."

mark finished
log "ALL EXPERIMENTS COMPLETE."
log "Results:"
ls -la results/ispd2005/*.json 2>&1 | tee -a "$RUN_DIR/pipeline.log"
