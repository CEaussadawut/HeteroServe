"""Tests for the scheduler review's items 2-4 (2026-11-01): predict_pool.PLANNER_V4 (a hybrid's memory as llama.cpp allocates
it, depth priced on every node) and energy_model's refit and per_node fix. stdlib unittest only.

    python3 -m unittest discover -s testbed/tests -q
"""
import json, os, sys, unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path[:0] = [os.path.join(ROOT, "testbed", "scripts"), os.path.join(ROOT, "ThinkPad")]
import predict_pool as P                                      # noqa: E402
import energy_model as EM                                     # noqa: E402

DS = os.path.join(ROOT, "testbed", "dataset")
SNAP = os.path.join(DS, "memory_snapshot_20261028_033438.json")     # def2's snapshot, the 35B's last serves
N4 = ["laptop", "pi5", "pi4", "jetson"]
MIB = 1048576.0
K35 = "qwen36moe_q6k"


class Base(unittest.TestCase):
    def setUp(self):
        self.env = os.environ.get("HETEROSERVE_RESERVE_MB")
        os.environ["HETEROSERVE_RESERVE_MB"] = ""
        self.mem = json.load(open(SNAP))
        self.shapes = json.load(open(os.path.join(DS, "model_shapes.json")))

    def tearDown(self):
        if self.env is None:
            os.environ.pop("HETEROSERVE_RESERVE_MB", None)
        else:
            os.environ["HETEROSERVE_RESERVE_MB"] = self.env


class HybridMemory(Base):
    def test_the_options(self):
        self.assertEqual(set(P.PLANNER_V4) - set(P.PLANNER_V3), {"hybrid_memory", "attn_table"})
        self.assertTrue(all(P.PLANNER_V4[k] == v for k, v in P.PLANNER_V3.items()))

    def test_the_35b_state_is_what_llama_cpp_logged(self):
        name = P.MODELS[K35][0]
        q = P.with_hybrid(self.shapes[name], name)
        st = P.state_bytes(q, 4096)
        self.assertEqual(len(st), 41)
        self.assertEqual(st[-1], 0.0)                                       # the MTP block: neither KV nor state
        kv = sum(s for i, s in enumerate(st) if (i + 1) % 4 == 0 and i < 40)
        self.assertAlmostEqual(kv / MIB, 80.0)                              # "size = 80.00 MiB (4096 cells, 10 layers"
        self.assertAlmostEqual((sum(st) - kv) / MIB, 62.8125)               # "size = 62.81 MiB", R 2.81 + S 60.00
        self.assertIsNone(P.state_bytes(self.shapes["qwen3-14b-q4_k_m"], 4096))   # a dense model: as before

    def test_the_gate_charges_each_device_its_logged_buffers(self):
        # the 35B's 28 Oct serve (hybrid_shapes_20261101.json check_35b): laptop 25 + Jetson 3 + Pi 4 4 + Pi 5 9, head on the Pi 5
        assign = [("laptop", 25), ("jetson", 3), ("pi4", 4), ("pi5", 9)]
        logged = {"laptop": 48.0 + 39.78, "jetson": 8.0 + 4.19, "pi4": 8.0 + 6.28, "pi5": 16.0 + 12.56}
        v4 = P.plan_fixed(K35, assign, "last", 4096, self.mem, load_mode="none", **P.PLANNER_V4)
        m4 = P.memory(v4["assign"], v4["q"], self.mem, v4["devices"], v4["model"], head_at=v4["head_at"])
        lb, start = v4["q"]["layer_bytes"], 0
        for n, k in assign:
            weights_only = P.memory_need_mb(v4["devices"][n], v4["model"], sum(lb[start:start + k]), k,
                                            holds_head=(n == "pi5"), holds_embd=(n == "laptop"),
                                            role="host" if n == "laptop" else "worker", uploads=(n == "laptop"))
            self.assertAlmostEqual(m4[n]["need_mib"] - weights_only, logged[n], delta=1.0, msg=n)
            start += k

    def test_the_laptop_takes_a_25th_block(self):
        v3 = P.plan_host_choice(K35, N4, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V3)
        v4 = P.plan_host_choice(K35, N4, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V4)
        self.assertEqual(dict(v3["assign"])["laptop"], 24)
        self.assertEqual(dict(v4["assign"])["laptop"], 25)
        self.assertLess(v4["decode_ms"], v3["decode_ms"] * 0.95)            # predicted 242.9 against 259.2 ms

    def test_a_dense_plan_is_unchanged(self):
        for key in ("qwen3_14b", "qwen3_32b_q6k", "qwen3moe_q6k"):
            a = P.plan_host_choice(key, N4, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V3)
            b = P.plan_host_choice(key, N4, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V4)
            self.assertEqual((a["assign"], a["head_at"]), (b["assign"], b["head_at"]), key)
            self.assertAlmostEqual(a["decode_ms"], b["decode_ms"], places=6)


class Depth(Base):
    def test_every_node_has_a_law_in_the_v4_table(self):
        tab = P.attention_table(P.ATTENTION_V4)
        q = self.shapes["qwen3-14b-q4_k_m"]
        for n in N4:
            self.assertIsNotNone(P.attn_ms(tab, n, q), n)
        self.assertAlmostEqual(tab["boards"]["pi5"]["ns_per_qhead_dim"], 24.605 / 1000 / (32 * 64 * 22) * 1e6, places=4)

    def test_the_old_tables_price_as_before(self):
        q = self.shapes["qwen3.6-35b-a3b-q6_k"]
        old = P.attention_table(P.ATTENTION_FA)
        self.assertIsNone(P.attn_ms(old, "pi5", q))                         # no Pi 5 law: depth planning with it failed
        lap = q["n_head"] * q["head_dim"] * old["boards"]["laptop"]["ns_per_qhead_dim"] * 1e-6
        self.assertAlmostEqual(P.attn_ms(old, "laptop", q), lap)            # the CPU law, no fraction, as before
        self.assertAlmostEqual(P.attn_ms(P.attention_table(P.ATTENTION_V4), "laptop", q), lap * 0.25)   # one block in four

    def test_depth_planning_with_the_pi5(self):
        r = P.plan_host_choice("qwen3moe_q6k", N4, 8192, self.mem, load_mode="none", depth=4096, **P.PLANNER_V4)
        self.assertIn("pi5", dict(r["assign"]))
        f = P.plan_fixed("qwen3moe_q6k", r["assign"], "host" if r["head_at"] == 0 else "last", 8192, self.mem,
                         load_mode="none", **P.PLANNER_V4)                  # the same plan at depth 0 costs less
        self.assertLess(f["decode_ms"], r["decode_ms"])


class Energy(Base):
    def test_per_node_sums_to_predict_with_planner_v3(self):
        r = P.plan_host_choice(K35, N4, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V3)
        busy, xfer = EM.per_node(r["assign"], r["q"], r["C"], r["links"], head_at=r["head_at"], rpc_msg=P.RPC_MSG)
        self.assertAlmostEqual(sum(b["tg"] for b in busy.values()) + xfer["tg"], r["decode_ms"], places=6)

    def test_the_refit_reproduces_its_own_fit(self):
        ref = EM.refit_table()
        lap = ref["laptop"]
        p = next(x for x in ref["points"] if x["key"] == "qwen3_14b" and x["placement"] == "engine")   # the laptop alone
        w = EM.laptop_refit_w(lap, 1.0, p["T_s"], "host", False)
        self.assertAlmostEqual(w, lap["active_w"] + lap["per_token_j"] / p["T_s"])
        self.assertLess(abs(w / p["W"]["laptop"] - 1), 0.05)
        self.assertEqual(EM.laptop_refit_w(lap, 0.0, 0.5, "none", False), lap["idle_w"])
        self.assertLess(abs(ref["fit"]["all_outlets_error"]["refit_leave_one_model_out"]["worst_pct"]), 10)

    def test_the_laptop_idle_moves_both_rows(self):
        r = P.plan_host_choice(K35, N4, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V4)
        kw = dict(head_at=r["head_at"], rpc_msg=P.RPC_MSG, t_ms=270.0)
        a = EM.energy_refit(r["assign"], r["q"], r["C"], r["links"], **kw)
        b = EM.energy_refit(r["assign"], r["q"], r["C"], r["links"], laptop_idle_w=17.1, **kw)
        shift = EM.refit_table()["laptop"]["idle_w"] - 17.1
        self.assertAlmostEqual(a["board_w"]["laptop"] - b["board_w"]["laptop"], shift, places=6)   # busy or idle alike
        self.assertEqual(a["board_w"]["pi5"], b["board_w"]["pi5"])

    def test_energy_refit_on_a_planner_v4_plan(self):
        r = P.plan_host_choice(K35, N4, 4096, self.mem, host="laptop", load_mode="none", **P.PLANNER_V4)
        e = EM.energy_refit(r["assign"], r["q"], r["C"], r["links"], head_at=r["head_at"], rpc_msg=P.RPC_MSG)
        self.assertAlmostEqual(e["T_pred_ms"], r["decode_ms"], places=6)
        self.assertAlmostEqual(e["plan_j"], e["all_j"])                     # every node in the plan
        # the old model now prices PLANNER_V3's scaled table too (its per_node check failed on it before 2026-11-01)
        old = EM.energy(r["assign"], r["q"], r["C"], r["links"], EM.power_table(), "tg", head_at=r["head_at"], t_ms=e["T_ms"])
        self.assertLess(old["gross_j"], e["plan_j"] * 0.9)                  # the laptop's old row: 14.05 W idle, 32.12 W busy
        self.assertGreater(e["board_w"]["laptop"], EM.refit_table()["laptop"]["idle_w"])


class Items9And10(unittest.TestCase):
    """2026-11-02: the request objective (item 9) and the split serve_pool.sh checks under MTP speculation (item 10)."""
    def test_a_request_is_priced_as_its_prefill_and_its_decode(self):
        self.assertEqual(P.objective_ms("decode", 150.0, 60.0), 150.0)
        self.assertEqual(P.objective_ms("prefill", 150.0, 60.0), 60.0)
        self.assertEqual(P.objective_ms({"prefill": 512, "decode": 64}, 150.0, 60.0), 512 * 60.0 + 64 * 150.0)
        self.assertEqual(P.objective_ms({"decode": 10}, 150.0, 60.0), 1500.0)

    def test_the_mtp_block_counts_on_the_last_node_only_with_mtp(self):
        q = {"layer_bytes": [10.0, 10.0, 10.0, 10.0, 7.0], "nextn_layers": 1, "head_bytes": 3.0}
        plan = [("laptop", 2), ("pi5", 1), ("pi4", 2)]
        self.assertEqual(P.planned_buffers(q, plan, "laptop"), {"laptop": 23.0, "pi5": 10.0, "pi4": 10.0})
        self.assertEqual(P.planned_buffers(q, plan, "laptop", mtp=True), {"laptop": 23.0, "pi5": 10.0, "pi4": 17.0})
        self.assertEqual(P.planned_buffers(dict(q, nextn_layers=0), plan, "pi4"), {"laptop": 20.0, "pi5": 10.0, "pi4": 20.0})
        self.assertEqual(P.planned_buffers({"head_bytes": 3.0}, plan, "laptop"), {"laptop": None, "pi5": None, "pi4": None})

    def test_serve_pool_checks_the_split_with_the_mtp_flag(self):
        src = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")).read()
        self.assertIn('mtp = "draft-mtp" in os.environ.get("SERVE_EXTRA", "") + " " + os.environ.get("SERVE_DRAFT", "")', src)
        self.assertIn("want = P.planned_buffers(q, assign, hn, mtp=mtp)", src)

    def test_the_jetsons_moe_prefill_multiplier(self):
        # 2026-11-03, jmoe: the Jetson's prefill per parameter on the 30B, scaled in both roles; heads and other boards as they were
        C = {"jetson": {"host": (1.0, 2.0), "worker": (1.0, 3.0), "worker_head": (5.0, 6.0)}, "pi5": {"worker": (1.0, 4.0)}}
        m = P.PREFILL_MULT["qwen3moe_q6k"]["jetson"]
        out = P.apply_prefill_mult(C, "qwen3moe_q6k", "measured")
        self.assertEqual(out["jetson"]["host"], (1.0, 2.0 * m))
        self.assertEqual(out["jetson"]["worker"], (1.0, 3.0 * m))
        self.assertEqual(out["jetson"]["worker_head"], (5.0, 6.0))
        self.assertEqual(out["pi5"], C["pi5"])
        self.assertIs(P.apply_prefill_mult(C, "qwen3moe_q6k", None), C)
        self.assertIs(P.apply_prefill_mult(C, "qwen36moe_q6k", "measured"), C)     # measured on the 30B only

    def test_serve_pool_refuses_mtp_with_the_mtp_block_on_the_pi5(self):
        # 2026-11-03: the guard's own lines, run with the plan's variables set (testbed/results/pi5_tx_stall_20261103)
        import subprocess
        lines = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")).read().split("\n")
        i = next(k for k, l in enumerate(lines) if l.startswith('  if [[ "$EXTRA $DRAFT_GIVEN" == *draft-mtp* ]]'))
        j = next(k for k in range(i, len(lines)) if lines[k] == "  fi")
        guard = "\n".join(lines[i:j + 1])

        def rc(extra, workers, force="", draft="none", tso_off=False):
            env = dict(os.environ, EXTRA=extra, DRAFT_GIVEN=draft, WORKERS=workers, PLAN="the plan", SERVE_MTP_PI5=force,
                       TSO_OFF="1" if tso_off else "0")
            # serve_pool.sh's on(): here it answers the Pi 5's TSO question from TSO_OFF instead of over ssh
            sh = ('on() { [ "$1" = pi5 ] && [ "$TSO_OFF" = 1 ]; }\n'
                  'EXTRA="$EXTRA"; DRAFT_GIVEN="$DRAFT_GIVEN"; WORKERS="$WORKERS"; PLAN="$PLAN"\n%s\nexit 0' % guard)
            return subprocess.run(["bash", "-c", sh], env=env, capture_output=True, text=True).returncode
        mtp = "--spec-type draft-mtp --spec-draft-n-max 2"
        self.assertEqual(rc(mtp, "jetson pi4 pi5"), 2)                 # the MTP block lands on the Pi 5: refused
        self.assertEqual(rc(mtp, "pi5"), 2)
        self.assertEqual(rc(mtp, "pi5 pi4 jetson"), 2)                 # the Pi 5 a middle worker: refused too (sp10d)
        self.assertEqual(rc(mtp, "pi4 jetson"), 0)                     # no Pi 5
        self.assertEqual(rc(mtp, ""), 0)                               # the laptop alone
        self.assertEqual(rc("", "jetson pi4 pi5"), 0)                  # no MTP
        self.assertEqual(rc(mtp, "jetson pi4 pi5", force="1"), 0)      # asked to try anyway
        self.assertEqual(rc("", "pi5", draft="--spec-type draft-mtp"), 2)   # MTP through a custom SERVE_DRAFT
        self.assertEqual(rc(mtp, "jetson pi4 pi5", tso_off=True), 0)   # the Pi 5 reports its TSO off: allowed (22:00 fix)
        self.assertEqual(rc(mtp, "pi5 pi4", tso_off=True), 0)

    def test_serve_pool_serves_mtp1_by_default_unless_the_pi5s_tso_is_on(self):
        # 2026-11-04 (the owner: "Turn MTP 1 on"): the planner block's flag, and the shell's block run with the plan's variables set
        import subprocess
        src = open(os.path.join(ROOT, "testbed", "scripts", "serve_pool.sh")).read()
        self.assertIn('SERVE_MTP_MODE=$(mtp_mode "${SERVE_MTP-__unset__}" "$MODEL" "$INTERACTIVE" "$DRAFT_GIVEN" "$EXTRA")', src)
        self.assertIn('auto = os.environ.get("SERVE_MTP_MODE") == "on" and not mtp and len(assign) > 1 '
                      'and int(q.get("nextn_layers") or 0) > 0', src)                # pooled, and the model has the head
        self.assertLess(src.index("mtp = mtp or auto"), src.index("want = P.planned_buffers(q, assign, hn, mtp=mtp)"))
        lines = src.split("\n")
        i = lines.index('  if [ "$MTP_AUTO" = 1 ]; then')
        j = next(k for k in range(i, len(lines)) if lines[k] == "  fi")
        self.assertLess(j, next(k for k, l in enumerate(lines) if l.startswith('  if [[ "$EXTRA $DRAFT_GIVEN" == *draft-mtp* ]]')))
        block = "\n".join(lines[i:j + 1])

        def run(auto, workers, tso_off=False, extra=""):
            # serve_pool.sh's on(): here it answers the Pi 5's TSO question from TSO_OFF instead of over ssh
            sh = ('. testbed/scripts/serve_defaults.sh\n'
                  'on() { [ "$1" = pi5 ] && [ "$TSO_OFF" = 1 ]; }\n'
                  'MODEL=qwen35_9b; MTP_AUTO=%s; WORKERS="%s"; EXTRA="%s"\n'
                  'declare -A WANT_MIB=([pi5]=10.0 [pi4]=17.0); declare -A WANT_MIB_NOMTP=([pi5]=10.0 [pi4]=10.0)\n'
                  '%s\necho "EXTRA=$EXTRA"; echo "PI4=${WANT_MIB[pi4]}"' % (auto, workers, extra, block))
            out = subprocess.run(["bash", "-c", sh], cwd=ROOT, env=dict(os.environ, TSO_OFF="1" if tso_off else "0"),
                                 capture_output=True, text=True, check=True).stdout
            return dict(l.split("=", 1) for l in out.splitlines() if l.startswith(("EXTRA=", "PI4=")))
        mtp1 = "--spec-type draft-mtp --spec-draft-n-max 1"
        self.assertEqual(run(1, "pi5 pi4", tso_off=True), {"EXTRA": mtp1, "PI4": "17.0"})    # served with MTP 1 and its block
        self.assertEqual(run(1, "pi4 jetson"), {"EXTRA": mtp1, "PI4": "17.0"})               # no Pi 5: its TSO not asked
        self.assertEqual(run(1, "pi5 pi4"), {"EXTRA": "", "PI4": "10.0"})                    # its TSO on: served without
        self.assertEqual(run(1, "pi4", extra="--backend-sampling"), {"EXTRA": mtp1 + " --backend-sampling", "PI4": "17.0"})
        self.assertEqual(run(0, "pi5 pi4"), {"EXTRA": "", "PI4": "17.0"})                    # not the default's case


if __name__ == "__main__":
    unittest.main()
