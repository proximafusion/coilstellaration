# Dev container

Open the repo in VS Code and run **Dev Containers: Reopen in Container**.

The first launch installs `hatch`, creates the `default`, `test`, and `lint`
environments, and installs the `pre-commit` git hooks. The container's
`/home/vscode` is persisted in a Docker volume across rebuilds, and
`~/tmp/outputs` on the host is bind-mounted into the container.
