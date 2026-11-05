#!/usr/bin/env python3
"""Summarise an rpc_smoke_pi5_jetson.sh log: per-cell throughput from the raw `-oe jsonl` rows, and the
RPC-minus-local comparison the script deliberately leaves to afterwards.

Throughput: mean and min-max of each cell's repetitions (samples_ts). They are the repetitions of ONE
llama-bench invocation, not separate runs -- the across-run drift is what the local1 -> local2 bracket
measures, and every RPC-vs-local difference is printed next to it.

Only cells the script itself marked `CELL <name>: ok` are used; any other cell is listed and left out.

    python3 testbed/scripts/summarise_rpc_smoke.py testbed/results/rpc_smoke_pi5_jetson_<stamp>.log
"""
import json
import re
import statistics as st
import sys


def main(path):
    cells, ok, cur = {}, set(), None
    for line in open(path, encoding='utf-8', errors='replace'):
        m = re.match(r'##### (\S+)\s*$', line)
        if m:
            cur = m.group(1)
            cells[cur] = {}
            continue
        m = re.match(r'CELL (\S+): (\S+)', line)
        if m:
            if m.group(2) == 'ok':
                ok.add(m.group(1))
            continue
        if cur and line.startswith('{"build_commit"'):
            r = json.loads(line)
            test = f"pp{r['n_prompt']}" if r['n_prompt'] else f"tg{r['n_gen']}"
            cells[cur][test] = r

    print(f'log: {path}')
    bad = [c for c in cells if c not in ok]
    if bad:
        print(f'NOT ok (left out): {", ".join(bad)}')
    print(f'\n{"cell":8} {"test":5} {"mean t/s":>9}  {"min-max":>17}  n  backend  ngl  commit')
    stats = {}
    for c, tests in cells.items():
        if c not in ok:
            continue
        for t, r in sorted(tests.items()):
            s = r['samples_ts']
            stats[(c, t)] = st.mean(s)
            print(f'{c:8} {t:5} {st.mean(s):9.3f}  [{min(s):7.3f}-{max(s):7.3f}]  {len(s)}  {r["backends"]:7}  {r["n_gpu_layers"]:3}  {r["build_commit"]}')

    tests = sorted({t for (_, t) in stats})
    if not all((c, t) in stats for c in ('local1', 'local2') for t in tests):
        print('\nno complete local1/local2 bracket -- no comparison')
        return
    print('\nlocal bracket (drift across the run) and RPC cells against the mean of local1 and local2:')
    for t in tests:
        l1, l2 = stats[('local1', t)], stats[('local2', t)]
        base = (l1 + l2) / 2
        print(f'  {t}: local2 vs local1 {100 * (l2 - l1) / l1:+.2f}%   (local mean {base:.3f} t/s)')
        for c in cells:
            if c.startswith('ngl=') and (c, t) in stats:
                v = stats[(c, t)]
                print(f'  {t}: {c:7} {v:8.3f} t/s = {100 * (v - base) / base:+.2f}% vs local')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
