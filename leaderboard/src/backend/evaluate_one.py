import io
import json
import logging
import os
import sys

import pandas as pd

from leaderboard.src.backend import runner

logger = logging.getLogger(__name__)


def main() -> None:
    req_path, out_path = sys.argv[1], sys.argv[2]
    with open(req_path) as f:
        req = json.load(f)
    logger.info(
        "Eval subprocess starting for model=%s revision=%s",
        req.get("model"),
        req.get("revision"),
    )
    record, rows, model_bytes = runner._run_eval(req)
    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)

    model_path = os.path.splitext(out_path)[0] + ".onnx"
    with open(model_path, "wb") as f:
        f.write(model_bytes)

    if rows:
        parquet_path = os.path.splitext(out_path)[0] + ".parquet"
        with open(parquet_path, "wb") as f:
            df = pd.DataFrame.from_records(rows) if rows else pd.DataFrame()
            df["results_json"] = out_path
            buf = io.BytesIO()
            df.to_parquet(buf, index=False)
            parquet_bytes = buf.getvalue()
            f.write(parquet_bytes)
        logger.info("Wrote %d raw instance rows to %s", len(rows), parquet_path)
    logger.info(
        "Eval subprocess done for model=%s: status=%s overall=%s -> %s",
        req.get("model"),
        record.get("status"),
        record.get("overall_score"),
        out_path,
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        force=True,
    )
    main()
