import io
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from leaderboard.src.backend import runner

logger = logging.getLogger(__name__)


def main() -> None:
    req_path, out_path = sys.argv[1], sys.argv[2]
    req = json.loads(Path(req_path).read_text())
    logger.info(
        "Eval subprocess starting for model=%s revision=%s",
        req.get("model"),
        req.get("revision"),
    )
    record, rows, model_bytes = runner._run_eval(req)
    Path(out_path).write_text(json.dumps(record, indent=2))
    Path(out_path).with_suffix(".onnx").write_bytes(model_bytes)

    if rows:
        parquet_path = Path(out_path).with_suffix(".parquet")
        df = pd.DataFrame.from_records(rows)
        df["results_json"] = out_path
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        parquet_path.write_bytes(buf.getvalue())
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
