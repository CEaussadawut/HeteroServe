# systemd units (control node)

    sudo install -m0644 heteroserve-collector@.service /etc/systemd/system/
    sudo systemctl daemon-reload
    sudo systemctl enable --now heteroserve-collector@$USER

The `@` in the filename makes it a template unit, so it runs as the user who owns
`/opt/heteroserve` and is in the `dialout` group - the collector needs serial access,
not root. That is also why the unit body says `User=%i`: the specifier only expands in
a template, and in a plain `.service` it would silently become an empty `User=`.

Leave the collector running for a whole measurement session. It appends one CSV per
session and the analysis cuts run windows out of it afterwards, so starting and stopping it
per run only risks losing the first seconds of a trace.

`iperf3 -s` is needed on this machine during link surveys. Run it from the harness
(`heteroctl survey` starts one if needed) or leave `iperf3 -s -D` running for the session.
