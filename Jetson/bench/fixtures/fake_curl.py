#!/usr/bin/env python3
"""Fake curl: serves $HS_SERVE/<basename> over file:// semantics, supports -I
(Content-Length), -o, and -C - (resume by appending from the current size)."""
import os, sys
a = sys.argv[1:]
serve = os.environ["HS_SERVE"]
url = [x for x in a if x.startswith("http")][-1]
name = url.rsplit("/", 1)[-1]
src = os.path.join(serve, name)
head = "-I" in a or "-sIL" in a or any(x.startswith("-sI") for x in a)
out = a[a.index("-o") + 1] if "-o" in a else None
resume = "-C" in a

if os.environ.get("HS_FAIL_URL") and os.environ["HS_FAIL_URL"] in url:
    sys.stderr.write("curl: (22) The requested URL returned error: 404\n")
    sys.exit(22)
if not os.path.exists(src):
    sys.stderr.write("curl: (22) 404\n")
    sys.exit(22)

if head:
    sys.stdout.write("HTTP/2 200\r\ncontent-length: %d\r\n\r\n" % os.path.getsize(src))
    sys.exit(0)

have = os.path.getsize(out) if (resume and out and os.path.exists(out)) else 0
total = os.path.getsize(src)
if have >= total:
    sys.stderr.write("curl: (33) HTTP server doesn't seem to support byte ranges\n")
    sys.exit(33)
with open(src, "rb") as fh:
    fh.seek(have)
    rest = fh.read()
with open(out, "ab" if have else "wb") as fh:
    fh.write(rest)
sys.exit(0)
