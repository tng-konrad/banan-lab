# Source from papers/emeans-rag-ivf before setup, live runs, or replay.
export EMEANS_WORKSPACE="/Users/kbanachewicz/Documents/banan-lab/papers/emeans-rag-ivf"
export TMPDIR="$EMEANS_WORKSPACE/tmp"
export XDG_CACHE_HOME="$EMEANS_WORKSPACE/cache/xdg"
export UV_CACHE_DIR="$EMEANS_WORKSPACE/cache/uv"
export UV_PYTHON_INSTALL_DIR="$EMEANS_WORKSPACE/cache/python"
export HF_HOME="$EMEANS_WORKSPACE/cache/huggingface"
export HF_HUB_CACHE="$EMEANS_WORKSPACE/cache/huggingface/hub"
export HF_DATASETS_CACHE="$EMEANS_WORKSPACE/cache/huggingface/datasets"
export TORCH_HOME="$EMEANS_WORKSPACE/cache/torch"
export MPLCONFIGDIR="$EMEANS_WORKSPACE/cache/matplotlib"
export JUPYTER_CONFIG_DIR="$EMEANS_WORKSPACE/cache/jupyter/config"
export JUPYTER_DATA_DIR="$EMEANS_WORKSPACE/cache/jupyter/data"
export JUPYTER_RUNTIME_DIR="$EMEANS_WORKSPACE/cache/jupyter/runtime"
export IPYTHONDIR="$EMEANS_WORKSPACE/cache/ipython"
export PYTHONPYCACHEPREFIX="$EMEANS_WORKSPACE/cache/pycache"
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export PYTHONUNBUFFERED=1
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$HF_HOME" "$MPLCONFIGDIR" \
         "$JUPYTER_CONFIG_DIR" "$JUPYTER_DATA_DIR" "$JUPYTER_RUNTIME_DIR" "$IPYTHONDIR" "$PYTHONPYCACHEPREFIX"
