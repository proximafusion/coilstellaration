from dataclasses import dataclass, make_dataclass


def fields(raw_class):
    return [v for k, v in raw_class.__dict__.items() if k[:2] != "__" and k[-2:] != "__"]


@dataclass(frozen=True)
class ColumnContent:
    name: str
    type: str
    displayed_by_default: bool
    hidden: bool = False
    never_hidden: bool = False


## Leaderboard columns
auto_eval_column_dict = []
auto_eval_column_dict.append(["model", ColumnContent, ColumnContent("Model", "markdown", True, never_hidden=True)])
auto_eval_column_dict.append(["submitter", ColumnContent, ColumnContent("Submitter", "str", True)])
auto_eval_column_dict.append(["overall", ColumnContent, ColumnContent("Overall ⬆️", "number", True)])
# fixed_shape track
auto_eval_column_dict.append(["fixed_mean", ColumnContent, ColumnContent("fixed: mean", "number", True)])
auto_eval_column_dict.append(["fixed_feas09", ColumnContent, ColumnContent("fixed: feas@0.9", "number", True)])
auto_eval_column_dict.append(["fixed_strict", ColumnContent, ColumnContent("fixed: strict", "number", True)])
auto_eval_column_dict.append(["fixed_median", ColumnContent, ColumnContent("fixed: median", "number", False)])
auto_eval_column_dict.append(["fixed_p10", ColumnContent, ColumnContent("fixed: p10", "number", False)])
# variable_shape track
auto_eval_column_dict.append(["var_mean", ColumnContent, ColumnContent("var: mean", "number", True)])
auto_eval_column_dict.append(["var_feas09", ColumnContent, ColumnContent("var: feas@0.9", "number", True)])
auto_eval_column_dict.append(["var_strict", ColumnContent, ColumnContent("var: strict", "number", True)])
auto_eval_column_dict.append(["var_median", ColumnContent, ColumnContent("var: median", "number", False)])
auto_eval_column_dict.append(["var_p10", ColumnContent, ColumnContent("var: p10", "number", False)])
# metadata
auto_eval_column_dict.append(["stochastic", ColumnContent, ColumnContent("Stochastic (K)", "str", True)])
auto_eval_column_dict.append(["submitted", ColumnContent, ColumnContent("Submitted", "str", False)])
auto_eval_column_dict.append(["revision", ColumnContent, ColumnContent("Revision", "str", False)])

AutoEvalColumn = make_dataclass("AutoEvalColumn", auto_eval_column_dict, frozen=True)


## Queue columns (submission tab)
@dataclass(frozen=True)
class EvalQueueColumn:
    model = ColumnContent("model", "markdown", True)
    revision = ColumnContent("revision", "str", True)
    submitter = ColumnContent("submitter", "str", True)
    stochastic = ColumnContent("stochastic", "str", True)
    status = ColumnContent("status", "str", True)


# Column selection
COLS = [c.name for c in fields(AutoEvalColumn) if not c.hidden]
EVAL_COLS = [c.name for c in fields(EvalQueueColumn)]
EVAL_TYPES = [c.type for c in fields(EvalQueueColumn)]

# Track-mean columns must be present for a row to display.
BENCHMARK_COLS = [
    AutoEvalColumn.fixed_mean.name,
    AutoEvalColumn.var_mean.name,
]
