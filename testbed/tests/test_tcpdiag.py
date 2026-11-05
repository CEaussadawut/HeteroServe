"""Tests for score_tcpdiag_20261009.py's parsing and its hypothesis, on ss output in both boards' formats, stdlib unittest.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "testbed", "scripts"))
import score_tcpdiag_20261009 as S                            # noqa: E402

JET = ["T 100.00",                                          # the Jetson: iproute2-ss180129 on 4.9, unacked/retrans only when set
       "Recv-Q    Send-Q           Local Address:Port            Peer Address:Port",
       "0         0                192.168.1.114:50052          192.168.1.110:41234",
       "\t cubic wscale:10,7 rto:208 rtt:4.602/7.02 ato:40 mss:1448 cwnd:10 ssthresh:21 bytes_acked:2461 send 25.2Mbps "
       "lastsnd:4 lastrcv:12 lastack:4 pacing_rate 30.2Mbps delivery_rate 88.4Mbps rcv_space:28960 rcv_ssthresh:40784 minrtt:0.131",
       "T 100.05",
       "Recv-Q    Send-Q           Local Address:Port            Peer Address:Port",
       "0         86000            192.168.1.114:50052          192.168.1.110:41234",
       "\t cubic wscale:10,7 rto:236 rtt:31.5/12.0 mss:1448 cwnd:88 ssthresh:60 bytes_acked:902461 send 32.2Mbps lastsnd:1 "
       "unacked:59 retrans:0/3 delivery_rate 19.8Mbps rcv_space:28960 rcv_ssthresh:40784 minrtt:0.131",
       "T 100.10",
       "Recv-Q    Send-Q           Local Address:Port            Peer Address:Port"]          # no connection in this sample
PI = ["T 100.02",                                            # the Pi 4: iproute2-6.15 on 6.18, bps units, rcv_wnd
      "Recv-Q Send-Q Local Address:Port  Peer Address:Port",
      "0      0      192.168.1.110:41234   192.168.1.114:50052",
      "\t cubic wscale:10,10 rto:204 rtt:1.193/1.396 mss:1448 cwnd:10 bytes_acked:4257 send 97099749bps lastsnd:8 "
      "delivery_rate 77226664bps delivered:15 app_limited busy:20ms rcv_space:14480 rcv_ssthresh:38912 minrtt:0.15 "
      "snd_wnd:64512 rcv_wnd:63488"]


class TestTcpdiag(unittest.TestCase):
    def test_both_formats_parse(self):
        j = S.parse_ss(JET)
        self.assertEqual(len(j), 2)                            # the empty sample adds nothing
        self.assertEqual((j[0]["rto"], j[0]["cwnd"], j[0]["unacked"], j[0]["retrans_total"]), (208, 10, 0, 0))
        self.assertEqual((j[1]["cwnd"], j[1]["unacked"], j[1]["retrans_total"], j[1]["srtt"]), (88, 59, 3, 31.5))
        self.assertAlmostEqual(j[1]["delivery_mbps"], 19.8)
        p = S.parse_ss(PI)
        self.assertEqual((p[0]["rcv_wnd"], p[0]["rcv_ssthresh"], p[0]["snd_wnd"]), (63488, 38912, 64512))
        self.assertAlmostEqual(p[0]["delivery_mbps"], 77.226664)

    def test_the_hypothesis(self):
        j, p = S.parse_ss(JET), S.parse_ss(PI)
        s = S.summarize(j, p, 158.0, (99.0, 101.0))
        self.assertEqual(s["jetson_cwnd_in_flight"], {"min": 88, "median": 88, "max": 88, "n": 1})
        self.assertEqual(s["jetson_retransmissions"], 3)
        self.assertEqual(S.td_ssai(s)["verdict"], "HOLDS")        # 158 ms < the median RTO 222 ms, in-flight cwnd 88 > 10
        self.assertEqual(S.td_ssai(S.summarize(j, p, 230.0, (99.0, 101.0)))["verdict"], "FAILS")   # a token longer than RTO
        j2 = [dict(x, cwnd=10) if x["unacked"] else x for x in j]
        self.assertEqual(S.td_ssai(S.summarize(j2, p, 158.0, (99.0, 101.0)))["verdict"], "FAILS")  # a restart window in flight
        self.assertEqual(S.td_ssai(S.summarize(j, p, 158.0, (200.0, 201.0)))["verdict"], "INCOMPLETE")  # no samples in window

    def test_the_decode_window_comes_from_llama_bench(self):
        line = json.dumps({"build_commit": "b23701f7", "n_prompt": 0, "n_gen": 64, "test_time": "2026-10-09T13:40:00Z",
                           "samples_ns": [10.1e9, 10.2e9, 10.0e9]})
        ms, t0, reps, ng = S.bench([line])
        self.assertAlmostEqual(ms, 10100.0 / 64)
        self.assertEqual((t0, reps, ng), (1791553200.0, 3, 64))   # 13:40:00Z
        a, b = S.window(ms, t0, reps, ng)
        self.assertAlmostEqual(b - a, 4 * 64 * ms / 1000 * 1.1 + 2.0, places=4)   # epochs near 1.8e9


if __name__ == "__main__":
    unittest.main()
