# shellcheck shell=bash
# https://google.github.io/oss-fuzz/getting-started/new-project-guide/#buildsh-script-environment

set -euo pipefail

python3 -m pip install .

find "$SRC" -maxdepth 1 \
  \( -name '*_seed_corpus.zip' -o -name '*.options' -o -name '*.dict' \) \
  -exec printf '[%s] Copying: %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" {} \; \
  -exec chmod a-x {} \; \
  -exec cp {} "$OUT" \;

# Build fuzzers in $OUT.
find "$SRC/jsonpickle/fuzzing" -name 'fuzz_*.py' -print0 | while IFS= read -r -d '' fuzz_harness; do
  compile_python_fuzzer "$fuzz_harness"
done
