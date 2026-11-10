def model_hyperlink(link, model_name):
    style = (
        "color: var(--link-text-color); "
        "text-decoration: underline;text-decoration-style: dotted;"
    )
    return f'<a target="_blank" href="{link}" style="{style}">{model_name}</a>'


def make_clickable_model(model_name):
    link = f"https://huggingface.co/{model_name}"
    return model_hyperlink(link, model_name)


def make_clickable_commit(model_name, commit_hash):
    if not commit_hash:
        return ""
    link = f"https://huggingface.co/{model_name}/commit/{commit_hash}"
    return model_hyperlink(link, commit_hash[:7])


def has_no_nan_values(df, columns):
    return df[columns].notna().all(axis=1)


def has_nan_values(df, columns):
    return df[columns].isna().any(axis=1)
