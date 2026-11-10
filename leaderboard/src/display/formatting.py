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


def make_user_link(username):
    """The username, linked to its Hugging Face profile."""
    if not username:
        return ""
    return model_hyperlink(f"https://huggingface.co/{username}", username)


def make_results_link(results_repo, path):
    """A "link" to `path` in the `results_repo` dataset; blank without a path."""
    if not path:
        return ""
    link = f"https://huggingface.co/datasets/{results_repo}/blob/main/{path}"
    return model_hyperlink(link, "link")


def make_clickable_commit(model_name, commit_hash):
    if not commit_hash:
        return ""
    link = f"https://huggingface.co/{model_name}/commit/{commit_hash}"
    return model_hyperlink(link, commit_hash[:7])


def has_no_nan_values(df, columns):
    return df[columns].notna().all(axis=1)


def has_nan_values(df, columns):
    return df[columns].isna().any(axis=1)
