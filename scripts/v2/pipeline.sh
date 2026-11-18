#!/usr/bin/env bash
# V2 from screened data to an evaluated, exported model (phases C-E finish, F, G, H, I without publishing).
# Resumable: each finished stage leaves $ROOT/done.<stage>; rerun the script after an interruption
# (training resumes from last.pt). Log: $ROOT/pipeline.log.
#   bash scripts/v2/pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export TQDM_DISABLE=1 PYTHONPATH=src:scripts
ROOT=~/sg-data/v2
MC=$ROOT/mc H=$ROOT/hard F=$ROOT/features A=$ROOT/annotations CFG=$ROOT/config.json
RUN=~/sg-data/runs/pl-v2 REL=$ROOT/release
SG=~/.venvs/sg/bin/python PK=~/.venvs/parakeet/bin/python VX=~/.venvs/voxcpm/bin/python
log() { echo "[$(date '+%F %T')] $*" | tee -a $ROOT/pipeline.log; }
todo() { [ ! -f $ROOT/done.$1 ]; }
done_() { touch $ROOT/done.$1; log "done: $1"; }
quiet() { grep -v -i -E "warn|memcpy" || true; }

screen_root() {   # two-ASR screening of every clip of $1 not screened yet; $2: extra flag for decide
    $SG scripts/v2/screen.py sets --root $1 | tee -a $ROOT/pipeline.log
    $PK scripts/asrtest/run_asr.py --engine parakeet --sets $1/asr/sets.jsonl --batch 32 2>&1 | quiet | tail -1
    $SG scripts/asrtest/run_asr.py --engine whisper --sets $1/asr/sets.jsonl 2>&1 | quiet | tail -1
    $SG scripts/v2/screen.py decide --root $1 $2 2>&1 | quiet | tee -a $ROOT/pipeline.log
}

log "start"
while pgrep -f "python.*run_asr.py --engine whisper --sets $MC" >/dev/null || pgrep -f "python.*generate_voxcpm" >/dev/null; do sleep 60; done

if todo mc; then
    $SG scripts/v2/screen.py decide --root $MC --finalize 2>&1 | quiet | tee -a $ROOT/pipeline.log
    done_ mc
fi
if todo hard1; then
    screen_root $H ""
    done_ hard1
fi
if todo hard2; then
    $VX scripts/v2/generate_voxcpm.py --texts $H/texts.jsonl --reference $H/reference.json --root $H --retry --max-temp 76 \
        > $H/gen-retry.log 2>&1
    screen_root $H --finalize
    done_ hard2
fi
if todo prepare; then
    python3 scripts/v2/dedupe_hard.py $MC $H | tee -a $ROOT/pipeline.log
    rm -rf $F
    for split in train validation test; do
        inputs=($MC/manifest-$split.jsonl)
        [ -f $H/manifest-$split.jsonl ] && inputs+=($H/manifest-$split.jsonl)
        $SG -m tiny_voca.prepare --inputs "${inputs[@]}" --split $split --output $F --config $CFG 2>&1 | quiet | tail -1 | tee -a $ROOT/pipeline.log
    done
    done_ prepare
fi
if todo timing; then
    $SG scripts/calibrate_aligner_timing.py --features $F --split train --count 300 --config $CFG --output $ROOT/timing.json 2>&1 | quiet | tail -3
    done_ timing
fi
OFFSET=$(python3 -c "import json; print(json.load(open('$ROOT/timing.json'))['suggested_onset_offset_seconds'])")
if todo annotate_loose; then
    rm -rf $A
    $SG -m tiny_voca.annotate --features $F --output $A --dataset-revision v2 --config $CFG --onset-offset $OFFSET \
        --min-token-probability 0 --max-weak-phone-fraction 1 --min-vowel-voiced 0 2>&1 | quiet | tail -1 | cut -c1-1500 | tee -a $ROOT/pipeline.log
    cp $A/summary.json $ROOT/annotations-loose-summary.json
    done_ annotate_loose
fi
if todo annotate; then
    LIMITS=$(python3 scripts/v2/qc_limits.py $ROOT/annotations-loose-summary.json)
    log "QC limits: $LIMITS (onset offset $OFFSET)"
    $SG -m tiny_voca.annotate --features $F --output $A --dataset-revision v2 --config $CFG --onset-offset $OFFSET \
        $LIMITS --resume 2>&1 | quiet | tail -1 | cut -c1-1500 | tee -a $ROOT/pipeline.log
    done_ annotate
fi
if todo train; then
    resume=()
    [ -f $RUN/last.pt ] && resume=(--resume $RUN/last.pt)
    log "train ${resume[*]}"
    # From scratch (no --init-from): the V1 weights are CC-BY-NC.
    $SG -u -m tiny_voca.train --train $A/train.jsonl --validation $A/validation.jsonl --output $RUN --config $CFG \
        --steps 104000 --gan-start 8000 --gan-ratio 0.7 --teacher-final 0 \
        --max-seconds 400000 --max-temp 76 --no-preload --log-every 100 "${resume[@]}" >> $ROOT/train.log 2>&1
    [ -f $RUN/inference.pt ] || { log "training ended without inference.pt"; exit 1; }
    done_ train
fi
if todo evaluate; then
    rm -rf $ROOT/eval
    $SG -m tiny_voca.evaluate --checkpoint $RUN/inference.pt --annotations $A --output $ROOT/eval --tag v2 \
        --splits validation test --per-voice 150 2>&1 | quiet | tail -c 300
    E=$ROOT/eval-hard
    rm -rf $E
    $SG scripts/v2/eval_hard.py render --texts $H/texts.jsonl --out $E --system v2:$RUN/inference.pt:MC \
        --system v1_F:weights/inference.pt:F --system v1_M:weights/inference.pt:M 2>&1 | quiet | tail -1
    $PK scripts/asrtest/run_asr.py --engine parakeet --sets $E/sets.jsonl 2>&1 | quiet | tail -1
    $SG scripts/asrtest/run_asr.py --engine whisper --sets $E/sets.jsonl 2>&1 | quiet | tail -1
    $SG scripts/v2/eval_hard.py score --out $E --teacher $H | tee -a $ROOT/pipeline.log
    done_ evaluate
fi
if todo export; then
    mkdir -p $REL
    cp $RUN/inference.pt $REL/slyszeglosy-v2.pt
    $SG -m tiny_voca.export --checkpoint $REL/slyszeglosy-v2.pt --output $REL/onnx/slyszeglosy-v2.onnx 2>&1 | quiet | tail -1
    $SG -m tiny_voca.export --checkpoint $REL/slyszeglosy-v2.pt --output $REL/onnx/exact.onnx --external-noise 2>&1 | quiet | tail -1
    $SG scripts/check_onnx.py --checkpoint $REL/slyszeglosy-v2.pt --onnx $REL/onnx/exact.onnx --release $REL/onnx/slyszeglosy-v2.onnx \
        --output $REL/check 2>&1 | quiet | tail -12 | tee -a $ROOT/pipeline.log
    sha256sum $REL/slyszeglosy-v2.pt $REL/onnx/slyszeglosy-v2.onnx | tee -a $ROOT/pipeline.log
    done_ export
fi
log "finished"
