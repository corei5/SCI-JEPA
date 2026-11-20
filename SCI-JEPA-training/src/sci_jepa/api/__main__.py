"""
Serve a trained SCI-JEPA model over HTTP (FastAPI, first version, no authentication).

    sci-jepa api --model <model folder> [--index auto|none|<paper_a_ku_index_*.pt>] [--port 8000]
    # same as: python -m sci_jepa.api ...

<model folder>: written by training (train.save_model): config.json + model.safetensors
(+ best.safetensors). --index auto finds the KU index the model was trained on under
$PAPER_A_INDEX_DIR (else the index_dir in the model's config). The first start with a full corpus
computes one vector per paper in a single pass over the KU vectors and caches it next to the index
(or in --cache-dir); later starts read the cache. /v1/ready answers 503 until everything is loaded.

Interactive docs: http://<host>:<port>/v1/docs
"""
import argparse
import logging
import os


def main(argv=None):
    ap = argparse.ArgumentParser("sci-jepa api", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="trained model folder (config.json + model.safetensors)")
    ap.add_argument("--weights", default="model", choices=["model", "best"], help="end of training or best validation epoch")
    ap.add_argument("--index", default="auto", help="'auto', 'none' (model-only: consistency, edit-check) or an index .pt")
    ap.add_argument("--cache-dir", default=os.environ.get("SCIJEPA_API_CACHE"), help="where to cache the paper vectors")
    ap.add_argument("--full-texts", action="store_true",
                    help="load the full KU texts (side file of the index; large) instead of 240-character snippets")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:1")
    ap.add_argument("--shortlist", type=int, default=64, help="papers whose KUs are scored per request")
    ap.add_argument("--no-neighbours", action="store_true", help="encode the user's paper alone (no corpus context)")
    ap.add_argument("--version", default="1.0.0", help="model version shown as sci-jepa-<version>")
    ap.add_argument("--session-ttl", type=int, default=24 * 3600, help="seconds before a session and its text are deleted")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    g = ap.add_argument_group("robustness (sci_jepa/api/guard.py)")
    g.add_argument("--gpu-queue", type=int, default=16, help="requests waiting for the GPU before 503 overloaded")
    g.add_argument("--gpu-wait", type=float, default=30.0, help="seconds a request may wait for the GPU before 503")
    g.add_argument("--rate-gpu", type=int, default=60, help="model requests per minute and client (0 = no limit)")
    g.add_argument("--rate-read", type=int, default=600, help="other requests per minute and client (0 = no limit)")
    g.add_argument("--max-body", type=int, default=64 * 1024, help="largest request body in bytes (413 above)")
    g.add_argument("--max-sessions", type=int, default=10_000, help="open /v1/ask sessions held in memory")
    g.add_argument("--trust-proxy", action="store_true",
                   help="take the client address from X-Forwarded-For (only behind your own reverse proxy)")
    g.add_argument("--cors-origin", action="append", default=[], help="allowed browser origin (repeatable)")
    g.add_argument("--exit-on-cuda-fault", action="store_true",
                   help="exit with code 3 after a CUDA fault so a supervisor (the sbatch loop) restarts the server")
    a = ap.parse_args(argv)

    import uvicorn
    from sci_jepa.api.app import create_app
    from sci_jepa.api.engine import Engine
    from sci_jepa.api.guard import Limits
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    log = logging.getLogger("sci_jepa.api")
    eng = Engine(a.model, weights=a.weights, device=a.device, index=a.index, cache_dir=a.cache_dir,
                 full_texts=a.full_texts, shortlist_papers=a.shortlist, neighbours=not a.no_neighbours,
                 version=a.version, log=log.info)
    limits = Limits(max_body_bytes=a.max_body, gpu_queue=a.gpu_queue, gpu_wait_s=a.gpu_wait, rate_gpu=a.rate_gpu,
                    rate_read=a.rate_read, trust_proxy=a.trust_proxy, max_sessions=a.max_sessions,
                    exit_on_cuda_fault=a.exit_on_cuda_fault, cors_origins=tuple(a.cors_origin),
                    idempotency_ttl_s=a.session_ttl)
    # One worker process: the model and its GPU memory exist once. limit_concurrency makes uvicorn itself answer
    # 503 when far more connections are open than the GPU queue can take; idle keep-alive connections close fast.
    uvicorn.run(create_app(eng, session_ttl=a.session_ttl, limits=limits), host=a.host, port=a.port,
                log_level="info", workers=1, limit_concurrency=max(64, 4 * a.gpu_queue), timeout_keep_alive=5,
                timeout_graceful_shutdown=30, proxy_headers=a.trust_proxy,
                forwarded_allow_ips="*" if a.trust_proxy else None)


if __name__ == "__main__":
    main()
