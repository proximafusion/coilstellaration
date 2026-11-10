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


def _column(attr, name, type_, shown, hidden=False):
    auto_eval_column_dict.append(
        [attr, ColumnContent, ColumnContent(name, type_, shown, hidden=hidden)]
    )


# Shown by default, as on the ConStellaration benchmark: when it was submitted,
# who submitted it, the scores and a link to the full results. Column names are
# lower case.
_column("submitted", "submission time", "date", True)
# The submitter's Hugging Face username, linked to their profile.
_column("user", "user", "markdown", True)
_column("strict", "strict", "number", True)
_column("mean", "mean", "number", True)
# "Link" to the per-instance scores parquet in the results dataset.
_column("full_results", "full results", "markdown", True)

# Offered in the Columns selector.
_column("model", "model id", "str", False)
_column("feas09", "feas@0.9", "number", False)
# Standard error of `strict`, a proportion over n instances: sqrt(p(1-p)/n).
_column("strict_se", "strict SE", "number", False)
_column("median", "median", "number", False)
_column("p10", "p10", "number", False)
# Number of instances scored in the selected track and stratum.
_column("n", "n", "number", False)
# Values stored in the ONNX graph, read at submission. Blank for older results.
_column("n_parameters", "params", "number", False)
# Benchmark version the result was produced with; blank for older results.
_column("bench_version", "version", "str", False)
_column("revision", "revision", "str", False)
_column("commit", "commit", "markdown", False)
_column("onnx_file", "onnx file", "str", False)

# Never shown: the plain username, which the search matches instead of the user
# column's HTML link (whose markup would otherwise match every row).
_column("user_name", "user_name", "str", False, hidden=True)

AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


## Queue columns (submission tab)
@dataclass(frozen=True)
class EvalQueueColumn:
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    submitter = ColumnContent("submitter", "str", True)
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

# Metric columns, each of whose best (maximum) value is highlighted.
METRIC_COLS = [
    AutoEvalColumn.mean.name,
    AutoEvalColumn.feas09.name,
    AutoEvalColumn.strict.name,
    AutoEvalColumn.median.name,
    AutoEvalColumn.p10.name,
]

# Count columns, shown as whole numbers.
INTEGER_COLS = [AutoEvalColumn.n.name, AutoEvalColumn.n_parameters.name]

# Rows are ordered by strict feasibility, then mean, both descending.
SORT_COLS = [AutoEvalColumn.strict.name, AutoEvalColumn.mean.name]

# Columns of every leaderboard frame. Hidden columns are included, so the search
# can use them, but are never offered or displayed.
COLS = [c.name for c in fields(AutoEvalColumn)]
EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]
