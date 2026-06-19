import json
import logging
import sys

from src.backend import runner

logger = logging.getLogger(__name__)


def main() -> None:
    req_path, out_path = sys.argv[1], sys.argv[2]
    with open(req_path) as f:
        req = json.load(f)
    logger.info(
        "Eval subprocess starting for model=%s revision=%s",
        req.get("model"), req.get("revision"),
    )
    record = runner._run_eval(req)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    logger.info(
        "Eval subprocess done for model=%s: status=%s overall=%s -> %s",
        req.get("model"), record.get("status"), record.get("overall_score"), out_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", force=True)
    main()
