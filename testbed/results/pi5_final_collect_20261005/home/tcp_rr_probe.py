#!/usr/bin/env python3
"""A pooled decode step's traffic, without llama.cpp: request/response over one TCP connection, timed per exchange.

    python3 tcp_rr_probe.py serve  <bind_ip> <port> <resp_bytes>                  # on the worker (the Pi 4)
    python3 tcp_rr_probe.py client <server_ip> <port> <req_bytes> <resp_bytes> <n> # on the host (the Pi 5)

Why (2026-09-29 evening): the same Qwen3.5-9B split (Pi 5 host 20 + Pi 4 worker 13 + head) decoded 9.4 % slower today than on
2026-09-29 00:18, unsteady across repetitions, while prefill repeated within 0.1 %. Clock throttling was ruled out (the Pi 4
held 1,800 MHz to 82 C), background CPU is 0.67 %, and both links read 1 Gbps full duplex. But the Pi 4's kernel counts 19,710
TCP retransmission TIMEOUTS since boot (the Pi 5: 88), and every timeout stalls a connection for at least 200 ms. A decode step
in that split sends the hidden state to the Pi 4 and brings about 1 MB of logits back (4 bytes x 248,320 vocabulary). This
sends the same shapes over the same path and times every exchange, so a stall shows as an outlier, not as a smaller mean.
The client prints one JSON line: the exchange times' percentiles, how many exceeded 100 ms, and the throughput.
"""
import json, socket, statistics, sys, time


def recv_exact(s, n):
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    while got < n:
        k = s.recv_into(view[got:], n - got)
        if k == 0:
            raise ConnectionError("peer closed after %d of %d bytes" % (got, n))
        got += k
    return buf


def serve(bind, port, resp):
    ls = socket.socket(); ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    ls.bind((bind, port)); ls.listen(1)
    c, _ = ls.accept(); c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    payload = bytes(resp)
    hdr = recv_exact(c, 8)
    req = int.from_bytes(hdr, "little")
    try:
        while True:
            recv_exact(c, req)
            c.sendall(payload)
    except ConnectionError:
        pass
    c.close(); ls.close()


def client(host, port, req, resp, n):
    s = socket.create_connection((host, port), timeout=30); s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.sendall(req.to_bytes(8, "little"))
    payload, dts = bytes(req), []
    t_all = time.perf_counter()
    for _ in range(n):
        t0 = time.perf_counter()
        s.sendall(payload)
        recv_exact(s, resp)
        dts.append(1000 * (time.perf_counter() - t0))
    wall = time.perf_counter() - t_all
    s.close()
    q = sorted(dts)
    pct = lambda p: q[min(len(q) - 1, int(p / 100 * len(q)))]
    print(json.dumps({"n": n, "req_bytes": req, "resp_bytes": resp, "ms_p50": round(pct(50), 3), "ms_p95": round(pct(95), 3),
                      "ms_p99": round(pct(99), 3), "ms_max": round(q[-1], 3), "ms_mean": round(statistics.fmean(q), 3),
                      "over_100ms": sum(d > 100 for d in q), "over_200ms": sum(d > 200 for d in q),
                      "MB_per_s": round(n * resp / wall / 1e6, 1), "wall_s": round(wall, 2)}))


if __name__ == "__main__":
    if sys.argv[1] == "serve":
        serve(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    else:
        client(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))
