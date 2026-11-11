"""A result's ONNX graph, shown in Netron.

As in the onnx-community Model Explorer Space, Netron (netron.app) runs in the
browser and fetches the ONNX file straight from the Hub, so the leaderboard
never downloads or parses the model. The file is the copy the evaluator stored
in the results dataset, next to the result's JSON summary.
"""

from html import escape
from pathlib import PurePosixPath
from urllib.parse import quote

from leaderboard.src.envs import RESULTS_REPO

NETRON_URL = "https://netron.app/"


def onnx_url(raw_scores_file: str) -> str:
    """The Hub download URL of the result's stored ONNX file."""
    path = PurePosixPath(raw_scores_file).with_suffix(".onnx").as_posix()
    return f"https://huggingface.co/datasets/{RESULTS_REPO}/resolve/main/{path}"


def netron_url(raw_scores_file: str) -> str:
    """The Netron page showing the result's ONNX graph."""
    return f"{NETRON_URL}?url={quote(onnx_url(raw_scores_file), safe=':/')}"


def viewer_html(raw_scores_file: str) -> str:
    """Netron in a frame, with a link to open it full size."""
    url = escape(netron_url(raw_scores_file))
    return (
        '<div class="pf-netron">'
        f'<iframe src="{url}" title="ONNX graph in Netron" loading="lazy"></iframe>'
        f'<p><a href="{url}" target="_blank" rel="noopener noreferrer">'
        "Open in Netron</a> · "
        f'<a href="{escape(onnx_url(raw_scores_file))}">Download the ONNX file</a>'
        "</p></div>"
    )
