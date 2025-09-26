#!/usr/bin/env bash
# check_tpm.sh - Query Groq OpenAI-compatible endpoint for x-ratelimit headers per model
# Usage:
#   ./check_tpm.sh                # uses default models list
#   ./check_tpm.sh model1 model2  # check only the models you pass

set -euo pipefail

# If GROQ_API_KEY not set, prompt (securely) so it doesn't end up in shell history
if [ -z "${GROQ_API_KEY-}" ]; then
  read -s -p "Enter Groq API key (will not echo): " GROQ_API_KEY
  echo
  export GROQ_API_KEY
fi

# Default list — change or pass your own model IDs as arguments
if [ $# -gt 0 ]; then
  models=("$@")
else
  models=(
    "openai/gpt-oss-120b"
    "llama-3.3-70b-versatile"
    "qwen-2.5-32b"
    "deepseek-r1-distill-llama-70b"
    "mistral-saba-24b"
  )
fi

endpoint="https://api.groq.com/openai/v1/chat/completions"

for m in "${models[@]}"; do
  echo "================================================================"
  echo "Model: $m"
  # Send minimal payload; -D - prints headers, -o /dev/null discards body
  # -s quiets curl progress; exit code still indicates network / HTTP issues
  curl -s -D - -o /dev/null -X POST "$endpoint" \
    -H "Authorization: Bearer $GROQ_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$m\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":1}" \
  | tr -d '\r' \
  | awk '
      BEGIN { found=0 }
      /^HTTP\//     { http=$0; print http }
      /^x-ratelimit-/ { print $0; found=1 }
      END { if (!found) print "Note: no x-ratelimit-* headers found for this response." }
    '
  echo
done

echo "Done."
