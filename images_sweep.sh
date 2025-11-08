#!/usr/bin/env bash
set -euo pipefail

LR_SET=("1e-5" "5e-5" "1e-4" "2e-4" "5e-4" "1e-3")
TRAIN_STEPS="1000"
DATASET="@me-dataset.zip"
CONST_ARGS=""

GO_FAST="false"
STEPS_INFER="35"
OUT_FMT="webp"

PROMPTS=(
  "Studio portrait of Sakib smiling in soft lighting"
  "Sakib wearing a leather jacket on a neon city street"
  "Cinematic close-up portrait of Sakib in golden hour light"
  "Sakib in futuristic cyberpunk armor with blue rim lighting"
)
PLABELS=("soft-smile_studio" "leather-neon" "golden-hour_closeup_ar3x4" "cyberpunk-armor")
SEEDS=("111" "222" "333" "444")

ROOT="IMAGES_BY_LR"
FLAT="${ROOT}/00_ALL_IMAGES_FLAT"
mkdir -p "${ROOT}" "${FLAT}"

safe_rm() {
  local target="$1"
  if [[ -e "${target}" || -L "${target}" ]]; then
    sudo rm -rf "${target}"
  fi
}

gpu_cleanup() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null \
    | awk -F, '/python|python3|cog/ {gsub(/ /,"",$1); print $1}' \
    | xargs -r -I{} bash -lc 'kill -TERM {} || true; sleep 2; kill -KILL {} || true'
  fi
}

gpu_wait_clear() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    for _ in $(seq 1 40); do
      if ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader 2>/dev/null | grep -Eiq 'python|cog'; then
        return 0
      fi
      sleep 3
    done
  fi
  return 0
}

predict_one() {
  local prompt="$1" seed="$2" zip_path="$3" outpath="$4" idx="$5"
  local status=0
  if [[ "$idx" -eq 2 ]]; then
    cog predict \
      -i prompt="${prompt}" \
      -i replicate_weights=@"${zip_path}" \
      -i go_fast="${GO_FAST}" \
      -i num_inference_steps="${STEPS_INFER}" \
      -i aspect_ratio="3:4" \
      -i output_format="${OUT_FMT}" \
      -i seed="${seed}" \
      -o "${outpath}" || status=$?
  else
    cog predict \
      -i prompt="${prompt}" \
      -i replicate_weights=@"${zip_path}" \
      -i go_fast="${GO_FAST}" \
      -i num_inference_steps="${STEPS_INFER}" \
      -i output_format="${OUT_FMT}" \
      -i seed="${seed}" \
      -o "${outpath}" || status=$?
  fi
  return $status
}

safe_rm output

for LR in "${LR_SET[@]}"; do
  RUN_DIR="${ROOT}/lr-${LR}"
  mkdir -p "${RUN_DIR}"

  if [[ -f "${RUN_DIR}/04_cyberpunk-armor__seed444.webp" ]]; then
    echo "\n=== LR ${LR} already completed, skipping ==="
    continue
  fi

  echo "\n=== LR ${LR} ==="
  echo "[CLEANUP] Clearing GPU before training..."
  gpu_cleanup || true
  gpu_wait_clear || true

  echo "[TRAIN] lr=${LR} steps=${TRAIN_STEPS}"
  safe_rm output
  cog train \
    -i dataset=${DATASET} \
    -i learning_rate="${LR}" \
    -i steps="${TRAIN_STEPS}" \
    ${CONST_ARGS}

  LORA_PATH="$(sudo find output -type f -name 'lora.safetensors' -printf '%T@ %p\n' 2>/dev/null | sort -nr | head -n1 | awk '{print $2}')"
  if [[ -z "${LORA_PATH:-}" || ! -f "${LORA_PATH}" ]]; then
    echo "ERROR: could not locate lora.safetensors for lr=${LR}" >&2
    exit 1
  fi
  OUT_DIR="$(dirname "${LORA_PATH}")"

  TMPDIR="$(mktemp -d "/tmp/lr_${LR//[^a-zA-Z0-9]/}_XXXXXX")"
  TMPZIP="${TMPDIR}/weights.zip"
  (
    cd "${OUT_DIR}" && sudo zip -q -j "${TMPZIP}" lora.safetensors settings.txt config.yaml 2>/dev/null || sudo zip -q -j "${TMPZIP}" lora.safetensors
  )
  sudo chown "$USER":"$USER" "${TMPZIP}" 2>/dev/null || true

  for i in "${!PROMPTS[@]}"; do
    P="${PROMPTS[$i]}"
    L="${PLABELS[$i]}"
    S="${SEEDS[$i]}"
    ORD=$(printf "%02d" $((i+1)))

    OUTPATH="${RUN_DIR}/${ORD}_${L}__seed${S}.webp"
    predict_one "${P}" "${S}" "${TMPZIP}" "${OUTPATH}" "$i"

    ACTUAL_PREFIX="${OUTPATH%.webp}"
    ACTUAL_FILE="${ACTUAL_PREFIX}.0.webp"
    if [[ -f "${ACTUAL_FILE}" ]]; then
      mv "${ACTUAL_FILE}" "${OUTPATH}"
    fi

    if [[ ! -f "${OUTPATH}" ]]; then
      echo "ERROR: expected image ${OUTPATH} not found" >&2
      exit 1
    fi

    cp "${OUTPATH}" "${FLAT}/lr-${LR}__${ORD}_${L}__seed${S}.webp"
  done

  rm -rf "${TMPDIR}"
  safe_rm "${OUT_DIR}"
  safe_rm output

  echo "[CLEANUP] Clearing GPU after lr=${LR}..."
  gpu_cleanup || true
  gpu_wait_clear || true

done

echo "\nSweep complete. Images saved under '${ROOT}' and '${FLAT}'."
