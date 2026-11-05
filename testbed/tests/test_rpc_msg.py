"""The opt-in RPC message term (predict_pool.RPC_MSG, rpc_bytes(), predict(..., rpc_msg=...), 2026-10-24): off by default; it
reproduces every byte figure rpc4, rpc5 and rpc6 measured with the laptop's TCP counters; and with it the predictor prices the
shaped arms of rpc4plan, ns4c and bs4 within their registered tolerance."""
import copy, json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import predict_pool as P  # noqa: E402

RES = os.path.join(ROOT, "testbed", "results")
DS = os.path.join(ROOT, "testbed", "dataset")
SHAPES = json.load(open(os.path.join(DS, "model_shapes.json")))
Q = SHAPES[P.MODELS["tinyllama_q4_0"][0]]


class RpcMsg(unittest.TestCase):
    def test_off_by_default(self):
        mem = json.load(open(os.path.join(DS, "memory_snapshot_20261024_142953.json")))
        e = P.plan_host_choice("tinyllama_q4_0", ["laptop", "pi4"], 2048, mem, host="laptop", load_mode="none")
        a = [("laptop", 11), ("pi4", 11)]
        self.assertEqual(P.predict(a, e["q"], e["C"], e["links"], head_at=0, rtt_trips=1),
                         P.predict(a, e["q"], e["C"], e["links"], head_at=0, rtt_trips=1, rpc_msg=None))

    def test_bytes_as_measured(self):
        h, lg = Q["activation_bytes"], 4 * 32000
        cases = [                      # (i, workers, out, depth, backend sampling) -> measured (to, back), B per token
            ((1, 1, h, 0, False), (19452, 8200)),       # rpc4 H; rpc5 H6, H11, H16; rpc6 C2, C4
            ((1, 1, lg, 0, False), (19452, 128008)),    # rpc4 L
            ((1, 1, lg, 0, True), (21369, 48)),         # rpc4 LB
            ((1, 2, h, 0, False), (19134, 8200)),       # rpc5 R2 Pi 5; R3 Pi 5
            ((2, 2, h, 0, False), (10946, 8200)),       # rpc5 R2 Pi 4; R3 Pi 4 (last of three)
            ((2, 3, h, 0, False), (10628, 8200)),       # rpc5 R3 Jetson (middle)
            ((1, 1, h, 850, False), (20988, 8200))]     # rpc6 D2, a cache of ~780-920 cells
        for (i, n, out, depth, bs), (to, back) in cases:
            inb, ob, _ = P.rpc_bytes(Q, i, n, out, depth, P.RPC_MSG, bs)
            self.assertEqual((inb, ob), (to, back), (i, n, out, depth, bs))

    def test_by_architecture(self):
        """rpc7 (Qwen3-32B dense; Qwen3.6-35B-A3B, whose rule was fitted on these data) and rpc8 (Qwen3-30B-A3B and
        Qwen3.5-9B, tested blind): every worker's bytes a token, from the laptop's TCP counters, to the byte."""
        cases = [("qwen3_32b_q6k", 1, 2, 43710), ("qwen3_32b_q6k", 2, 2, 23234),                       # rpc7 D32
                 ("qwen36moe_q6k", 1, 3, 28288), ("qwen36moe_q6k", 2, 3, 10958), ("qwen36moe_q6k", 3, 3, 11276),   # rpc7 M35
                 ("qwen3moe", 1, 2, 76824), ("qwen3moe", 2, 2, 10946),                                 # rpc8 M30
                 ("qwen35_9b", 1, 2, 35848), ("qwen35_9b", 2, 2, 19468)]                               # rpc8 H9
        for key, i, n, to in cases:
            q = SHAPES[P.MODELS[key][0]]
            inb, ob, _ = P.rpc_bytes(q, i, n, q["activation_bytes"], 0, P.RPC_MSG, False)
            self.assertEqual((inb, ob), (to, q["activation_bytes"] + 8), (key, i, n))

    def test_dense_term_kept(self):
        """RPC_MSG_DENSE is the evening's term: every model as dense (rpc7's registered predictions), and TinyLlama the same
        under both."""
        for key, i, n, to in (("qwen36moe_q6k", 1, 3, 19134), ("qwen36moe_q6k", 2, 3, 10628), ("qwen3moe", 1, 2, 19134),
                              ("qwen35_9b", 2, 2, 19138)):
            q = SHAPES[P.MODELS[key][0]]
            self.assertEqual(P.rpc_bytes(q, i, n, q["activation_bytes"], 0, P.RPC_MSG_DENSE, False)[0], to, key)
        for i, n in ((1, 1), (1, 2), (2, 2), (2, 3)):
            self.assertEqual(P.rpc_bytes(Q, i, n, Q["activation_bytes"], 0, P.RPC_MSG, False),
                             P.rpc_bytes(Q, i, n, Q["activation_bytes"], 0, P.RPC_MSG_DENSE, False))

    def test_no_dependence_on_blocks(self):
        self.assertNotIn("m_b", P.RPC_MSG)          # rpc5: 19,452 B at 6, 11 and 16 blocks

    def test_shaped_arms_priced(self):
        for sess, reg in (("rpc4plan_20261024_143036", "rpc4plan_predictions_20261024.json"),
                          ("ns4c_20261024_095925", "ns4c_predictions_20261024.json"),
                          ("bs4_20261024_104838", "bs4_predictions_20261024.json")):
            s, r = json.load(open(os.path.join(RES, sess, "score.json"))), json.load(open(os.path.join(DS, reg)))
            mem = json.load(open(os.path.join(ROOT, r["memfile"])))
            e = P.plan_host_choice("tinyllama_q4_0", ["laptop", "pi4"], 2048, mem, host="laptop", load_mode="none")
            for k, a in r["arms"].items():
                rate = a["setting"]["rate_mbit"]
                if not rate or a["setting"].get("delay_ms"):
                    continue
                links = copy.deepcopy(e["links"]); links["pi4"]["mbps"] = float(rate)
                bs = "SERVE_EXTRA" in a["serve_env"]
                d = P.predict([("laptop", 11), ("pi4", 11)], e["q"], e["C"], links, head_at=0 if a["head"] == "host" else None,
                              rtt_trips=1, rpc_msg=P.RPC_MSG, backend_sampling=bs)[0]
                m = s["arms"][k]["mean_ms_per_token"]
                self.assertLess(abs(m / d - 1), 0.06, "%s %s: measured %.1f, priced %.1f" % (sess, k, m, d))


class PlannerOptions(unittest.TestCase):
    """The RPC term and backend sampling as planner options (2026-10-25): off by default; on, they price plans as predict()
    does, and the planner's head choice on bs4's shaped link matches what bs4 measured (the head on the host, 146.31 ms,
    beat the head on the Pi 4 sampling there, 153.01 ms, at 5 Mbit/s)."""
    MEM = os.path.join(DS, "memory_snapshot_20261024_142953.json")

    def shaped(self, mbps):
        pairs = json.load(open(os.path.join(DS, "link.json")))["pairs"]
        return {"pi4-laptop": dict(pairs["pi4-laptop"], mbps=float(mbps))}

    def test_off_by_default(self):
        mem = json.load(open(self.MEM))
        a = P.plan_host_choice("tinyllama_q4_0", ["laptop", "pi4"], 2048, mem, host="laptop", load_mode="none", order_check=True)
        b = P.plan_host_choice("tinyllama_q4_0", ["laptop", "pi4"], 2048, mem, host="laptop", load_mode="none", order_check=True,
                               rpc_msg=None, backend_sampling=False)
        self.assertEqual((a["assign"], a["head_at"], a["decode_ms"]), (b["assign"], b["head_at"], b["decode_ms"]))

    def test_plan_fixed_prices_as_predict(self):
        mem = json.load(open(self.MEM))
        e = P.plan_fixed("tinyllama_q4_0", [("laptop", 11), ("pi4", 11)], "last", 2048, mem, load_mode="none",
                         extra_pairs=self.shaped(5), rtt_trips=1, rpc_msg=P.RPC_MSG, backend_sampling=True)
        d = P.predict([("laptop", 11), ("pi4", 11)], e["q"], e["C"], e["links"], head_at=None, rtt_trips=1,
                      rpc_msg=P.RPC_MSG, backend_sampling=True)[0]
        self.assertAlmostEqual(e["decode_ms"], d, places=6)

    def test_head_choice_on_bs4s_link(self):
        mem = json.load(open(self.MEM))
        kw = dict(host="laptop", load_mode="none", order_check=True, rtt_trips=1, extra_pairs=self.shaped(5),
                  rpc_msg=P.RPC_MSG, backend_sampling=True)
        e = P.plan_host_choice("tinyllama_q4_0", ["laptop", "pi4"], 2048, mem, **kw)
        on_host = P.plan_fixed("tinyllama_q4_0", [("laptop", 11), ("pi4", 11)], "host", 2048, mem, load_mode="none",
                               extra_pairs=self.shaped(5), rtt_trips=1, rpc_msg=P.RPC_MSG, backend_sampling=True)["decode_ms"]
        last = P.plan_fixed("tinyllama_q4_0", [("laptop", 11), ("pi4", 11)], "last", 2048, mem, load_mode="none",
                            extra_pairs=self.shaped(5), rtt_trips=1, rpc_msg=P.RPC_MSG, backend_sampling=True)["decode_ms"]
        self.assertLess(on_host, last)                       # as measured: 146.31 against 153.01 ms
        self.assertLessEqual(e["decode_ms"], min(on_host, last) + 1e-9)   # its pick is the fastest of them all
        if len(e["assign"]) > 1:
            self.assertEqual(e["head_at"], 0)                # a two-board pick keeps the head on the host
        else:
            self.assertEqual(e["assign"], [("laptop", 22)])  # at 5 Mbit/s: the laptop alone


if __name__ == "__main__":
    unittest.main()
