def model_hyperlink(link, model_name):
    style = (
        "color: var(--link-text-color); "
        "text-decoration: underline;text-decoration-style: dotted;"
    )
    return (
        f'<a target="_blank" rel="noopener noreferrer" href="{link}" '
        f'style="{style}">{model_name}</a>'
    )


def make_clickable_model(model_name):
    link = f"https://huggingface.co/{model_name}"
    return model_hyperlink(link, model_name)


def make_file_link(model_name, ref, onnx_path):
    """A "Link" to `onnx_path` in the `model_name` repo at `ref` (a commit) on the
    Hub."""
    link = f"https://huggingface.co/{model_name}/blob/{ref}/{onnx_path}"
    return model_hyperlink(link, "Link")


def make_clickable_commit(model_name, commit_hash):
    if not commit_hash:
        return ""
    link = f"https://huggingface.co/{model_name}/commit/{commit_hash}"
    return model_hyperlink(link, commit_hash[:7])


def has_no_nan_values(df, columns):
    return df[columns].notna().all(axis=1)


def has_nan_values(df, columns):
    return df[columns].isna().any(axis=1)
