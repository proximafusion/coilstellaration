from src.display.formatting import make_clickable_commit


def test_clickable_commit_shows_short_sha_and_links_to_commit():
    sha = "deadbeefcafebabe1234567890abcdef12345678"
    html = make_clickable_commit("org/model", sha)
    assert "deadbee" in html  # short 7-char SHA shown
    assert f"huggingface.co/org/model/commit/{sha}" in html
    assert html.startswith("<a")


def test_clickable_commit_empty_when_no_hash():
    assert make_clickable_commit("org/model", "") == ""
