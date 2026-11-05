#!/usr/bin/env python3
"""A live dashboard for an experiment in progress (topic-form scope 10: throughput, latency and power during experiments;
2026-10-08). Read-only and passive: it reads the run's own files on the laptop and never contacts a board or the strip, so
it cannot perturb a measurement.

    python3 testbed/scripts/dashboard.py [--run testbed/results/<run dir>] [--port 8765]

then open http://127.0.0.1:8765/ (bound to localhost only). Without --run it follows the newest run dir under
testbed/results that has a strip.csv or a run.jsonl (full_eval_*, energy_pool_*, netshape_*), re-checked on every refresh.
Shows, refreshed every 3 s:
- each outlet's power over the last 15 minutes (strip.csv, written by shelly_strip.py log; names from energy/strip.json);
- each served request's decode latency by placement: the median and P95 of the server's per-token gaps (run.jsonl's req
  lines, qr3_tpot.samples), and its time to first token;
- the run's last event, and per placement the requests so far and their median TPOT.
"""
import csv, glob, http.server, io, json, os, statistics, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import qr3_tpot as Q                                        # noqa: E402

WINDOW_S, MAX_POINTS, TAIL_BYTES = 900, 900, 8 << 20


def newest_run():
    c = [d for d in glob.glob(os.path.join(ROOT, "testbed", "results", "*")) if os.path.isdir(d)
         and (os.path.exists(os.path.join(d, "strip.csv")) or os.path.exists(os.path.join(d, "run.jsonl")))]
    return max(c, key=os.path.getmtime) if c else None


def outlet_names():
    try:
        m = json.load(open(os.path.join(ROOT, "testbed", "energy", "strip.json"))).get("outlets") or {}
        return {int(k): (v if isinstance(v, str) else v.get("label") or v.get("name") or str(k)) for k, v in m.items()}
    except (OSError, ValueError, AttributeError):
        return {}


def power(run, now=None):
    """{"t": [...], "outlets": {i: [W, ...]}} for the last WINDOW_S of the run's strip.csv, at most MAX_POINTS points."""
    p = os.path.join(run, "strip.csv")
    if not os.path.exists(p):
        return {"t": [], "outlets": {}}
    with open(p, "rb") as f:
        head = f.readline().decode()
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(len(head), size - TAIL_BYTES))
        body = f.read().decode("utf-8", "replace")
    if size > TAIL_BYTES + len(head):
        body = body.split("\n", 1)[1] if "\n" in body else ""
    rows = list(csv.DictReader(io.StringIO(head + body)))
    rows = [r for r in rows if r.get("ok") == "1" and r.get("wall_time")]
    if not rows:
        return {"t": [], "outlets": {}}
    end = float(rows[-1]["wall_time"]) if now is None else now
    rows = [r for r in rows if float(r["wall_time"]) >= end - WINDOW_S]
    step = max(1, -(-len(rows) // MAX_POINTS))                # ceiling: at most MAX_POINTS points
    rows = rows[::step]
    idx = sorted({int(k[1:].split("_")[0]) for k in rows[0] if k.startswith("o") and k.endswith("_apower")})
    out = {"t": [round(float(r["wall_time"]), 1) for r in rows], "outlets": {}}
    for i in idx:
        out["outlets"][i] = [float(r["o%d_apower" % i]) if r.get("o%d_apower" % i) not in (None, "") else None for r in rows]
    return out


def requests(run):
    """Per served request (run.jsonl): placement, round, i, sent, the server's median and P95 TPOT, TTFT; plus the last
    event and per-placement totals."""
    p = os.path.join(run, "run.jsonl")
    reqs, last = [], None
    lines = []
    if os.path.exists(p):
        with open(p, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    if lines:
        for ln in lines:
            try:
                r = json.loads(ln)
            except ValueError:
                continue                                    # a line still being written
            last = {"t": r["t"], "kind": r["kind"], "data": r["data"] if r["kind"] != "req" else
                    {k: r["data"].get(k) for k in ("placement", "round", "i")}}
            if r["kind"] != "req":
                continue
            x = r["data"]
            cg, sg, merged, tc, ts, ntok = Q.samples(x)
            if not sg:
                continue
            reqs.append({"t": round(x["sent_wall"], 1), "placement": x.get("placement"), "round": x.get("round"), "i": x.get("i"),
                         "tpot_p50": round(Q.pct(sg, 50), 1), "tpot_p95": round(Q.pct(sg, 95), 1),
                         "ttft_ms": round(tc, 1) if tc else None, "tokens": ntok})
    per = {}
    for q in reqs:
        per.setdefault(q["placement"], []).append(q["tpot_p50"])
    totals = {k: {"requests": len(v), "median_tpot_ms": round(statistics.median(v), 1)} for k, v in per.items()}
    return {"requests": reqs, "last": last, "per_placement": totals}


def data(run):
    run = run or newest_run()
    if not run:
        return {"run": None}
    return {"run": os.path.relpath(run, ROOT), "now": time.time(), "names": outlet_names(), "power": power(run),
            **requests(run)}


PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>HeteroServe live</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
:root{--bg:#fbfbfa;--fg:#1d1d1b;--mute:#6b6b66;--line:#e2e1dc}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mute:#9a9992;--line:#2c2c2a}}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:18px;margin:0 0 4px} .mute{color:var(--mute)} .grid{display:grid;gap:16px;grid-template-columns:1fr}
@media(min-width:900px){.grid{grid-template-columns:1fr 1fr}}
section{border:1px solid var(--line);border-radius:8px;padding:12px} canvas{width:100%!important;height:280px!important}
table{border-collapse:collapse;width:100%} td,th{border-bottom:1px solid var(--line);padding:4px 6px;text-align:left}
</style></head><body><main>
<h1>HeteroServe, live</h1><div class="mute" id="meta">loading…</div>
<div class="grid" style="margin-top:12px">
<section><b>Power per outlet (W), last 15 min</b><canvas id="pw"></canvas></section>
<section><b>Decode latency per request (server TPOT P50, ms)</b><canvas id="tp"></canvas></section>
</div>
<section style="margin-top:16px"><b>By placement</b><table id="tb"><thead><tr><th>placement</th><th>requests</th><th>median TPOT (ms)</th></tr></thead><tbody></tbody></table></section>
</main><script>
const C=["#2f6fb3","#b3562f","#3f8f4f","#8a5bb0","#a08a2a","#2a8a8a"];let pw,tp;
function hhmm(t){const d=new Date(t*1000);return d.toTimeString().slice(0,8)}
async function tick(){
 const r=await fetch("data.json");const d=await r.json();
 if(!d.run){document.getElementById("meta").textContent="no run directory yet";return}
 const last=d.last?(hhmm(d.last.t)+" "+d.last.kind+" "+JSON.stringify(d.last.data).slice(0,120)):"no events";
 document.getElementById("meta").textContent=d.run+" · last event: "+last;
 const P=d.power,ds=Object.keys(P.outlets).map((k,i)=>({label:(d.names[k]||("outlet "+k)),data:P.outlets[k],borderColor:C[i%6],pointRadius:0,borderWidth:1.5}));
 if(!pw){pw=new Chart(document.getElementById("pw"),{type:"line",data:{labels:P.t.map(hhmm),datasets:ds},options:{animation:false,scales:{y:{beginAtZero:true}}}})}
 else{pw.data.labels=P.t.map(hhmm);pw.data.datasets=ds;pw.update()}
 const pl=[...new Set(d.requests.map(q=>q.placement))];
 const ts=pl.map((p,i)=>({label:p,data:d.requests.filter(q=>q.placement===p).map(q=>({x:q.t,y:q.tpot_p50})),backgroundColor:C[i%6]}));
 if(!tp){tp=new Chart(document.getElementById("tp"),{type:"scatter",data:{datasets:ts},options:{animation:false,scales:{x:{ticks:{callback:v=>hhmm(v)}}}}})}
 else{tp.data.datasets=ts;tp.update()}
 document.querySelector("#tb tbody").innerHTML=Object.entries(d.per_placement).map(([k,v])=>`<tr><td>${k}</td><td>${v.requests}</td><td>${v.median_tpot_ms}</td></tr>`).join("");
}
tick();setInterval(tick,3000);
</script></body></html>"""


def main(argv):
    run = os.path.abspath(argv[argv.index("--run") + 1]) if "--run" in argv else None
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8765

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path.startswith("/data.json"):
                body, ctype = json.dumps(data(run)).encode(), "application/json"
            elif self.path in ("/", "/index.html"):
                body, ctype = PAGE.encode(), "text/html; charset=utf-8"
            else:
                self.send_error(404); return
            self.send_response(200); self.send_header("Content-Type", ctype); self.send_header("Cache-Control", "no-store")
            self.end_headers(); self.wfile.write(body)

        def log_message(self, *a):
            pass
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", port), H)
    print("dashboard on http://127.0.0.1:%d/ (run: %s)" % (port, os.path.relpath(run, ROOT) if run else "the newest"), flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main(sys.argv[1:])
