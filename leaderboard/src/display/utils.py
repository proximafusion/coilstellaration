from dataclasses import dataclass, make_dataclass


def fields(raw_class):
    return [
        v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"
    ]


@dataclass(frozen=True)
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False


## Leaderboard columns
# The table shows one track at a time (selected via the Track radio), so the
# metric columns are track-agnostic; the Track × Stratum selection decides which
# track's numbers fill them.
auto_eval_column_dict = []
auto_eval_column_dict.append(
    [
        "model",
        ColumnContent,
        ColumnContent("Model", "markdown", True, never_hidden=True),
    ]
)
auto_eval_column_dict.append(
    ["submitter", ColumnContent, ColumnContent("Submitter", "str", True)]
)
# per-track metrics
auto_eval_column_dict.append(
    ["mean", ColumnContent, ColumnContent("mean", "number", True)]
)
auto_eval_column_dict.append(
    ["feas09", ColumnContent, ColumnContent("feas@0.9", "number", False)]
)
auto_eval_column_dict.append(
    ["strict", ColumnContent, ColumnContent("strict", "number", True)]
)
auto_eval_column_dict.append(
    ["median", ColumnContent, ColumnContent("median", "number", False)]
)
auto_eval_column_dict.append(
    ["p10", ColumnContent, ColumnContent("p10", "number", False)]
)
# metadata
auto_eval_column_dict.append(
    ["stochastic", ColumnContent, ColumnContent("Stochastic (K)", "str", False)]
)
auto_eval_column_dict.append(
    ["submitted", ColumnContent, ColumnContent("Submitted", "str", True)]
)
auto_eval_column_dict.append(
    ["revision", ColumnContent, ColumnContent("Revision", "str", False)]
)
auto_eval_column_dict.append(
    ["commit", ColumnContent, ColumnContent("Commit", "markdown", True)]
)
auto_eval_column_dict.append(
    ["onnx_file", ColumnContent, ColumnContent("ONNX file", "str", True)]
)

AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


## Queue columns (submission tab)
@dataclass(frozen=True)
class EvalQueueColumn:
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    submitter = ColumnContent("submitter", "str", True)
    stochastic = ColumnContent("stochastic", "str", True)
    status = ColumnContent("status", "str", True)
    # Claim time (UTC ISO) — populated for RUNNING/FINISHED so a stuck job is visible.
    started = ColumnContent("started", "str", True)
    tracks = ColumnContent("tracks", "str", True)


# Tracks offered by the leaderboard's track selector, mapped to the radio label
# shown to the user.
TRACKS = ["fixed_shape", "variable_shape"]
TRACK_LABELS = {"fixed_shape": "fixed", "variable_shape": "variable"}

# Stratum views offered by the leaderboard's stratum selector. "All" is the
# all-strata pool (the paper's headline); the others slice a single stratum.
STRATUM_VIEWS = ["All", "tight", "medium", "loose"]

# Rows are ordered by strict feasibility, then mean, both descending.
SORT_COLS = [AutoEvalColumn.strict.name, AutoEvalColumn.mean.name]

# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]
