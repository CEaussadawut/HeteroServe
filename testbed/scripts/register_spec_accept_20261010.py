#!/usr/bin/env python3
"""Register scheduler upgrade 1, step 2 -- the draft model's acceptance -- before it runs. Writes, read-only,
testbed/dataset/spec_accept_predictions_20261010.json.

    python3 testbed/scripts/register_spec_accept_20261010.py [--dry-run]

What it measures: how often Qwen3-0.6B's drafted tokens are what Qwen3-14B would have produced, on 50 prompts fixed here
(40 English, 10 Thai; explanations, how-to, code, arithmetic, writing, technical questions), each answered in Qwen3's
non-thinking chat format, greedy, up to 128 tokens. Acceptance is a property of the two models, the prompts and the
decoding, not of the hardware, so it runs on the laptop (both models from its verified cache, the pinned llama.cpp) and
leaves the boards free; float differences between the laptop's CPU and the boards can flip a near-tie, and step 3 measures
acceptance again on the boards. llama-server drafts 4 tokens per check (--spec-draft-n-max 4, --spec-draft-p-min 0) and
logs each check ("accepted n/4 draft tokens"). With greedy drafting, a check of d < 4 drafted tokens accepts min(L, d) of
them, L the accepted run of the 4-token check, so one run gives tau(d) = 1 + E[min(L, d)] for d = 1..4.

Hypothesis, decided before any data, from step 1's measured pass times (spec_pass_20261010, T(2) / T(1) = 1.228):
  SA-PAY1   alpha1, the share of 4-token checks whose first drafted token is accepted, is at least 0.60: the level at
            which one drafted token per check reaches the survey's 1.3x with a free drafter ((1 + alpha1) / 1.228 >= 1.3).
Predicted (the survey's assumption): alpha = 0.7 at every position. Exploratory: the acceptance per position, tau(d), the
ceiling speedup per d with step 1's T(k), English against Thai, and the laptop's own speeds (not used for the boards).
"""
import datetime, hashlib, json, os, stat, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DS = os.path.join(ROOT, "testbed", "dataset")
OUT = os.path.join(DS, "spec_accept_predictions_20261010.json")
CACHE = os.path.expanduser("~/.cache/heteroserve-models")
TARGET = ("Qwen3-14B-Q4_K_M.gguf", "500a8806e85ee9c83f3ae08420295592451379b4f8cf2d0f41c15dffeb6b81f0")
DRAFT = ("Qwen3-0.6B-Q8_0.gguf", "9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031")
SERVER = "/opt/heteroserve/llama.cpp/build/bin/llama-server"
STEP1 = os.path.join(ROOT, "testbed", "results", "spec_pass_20261010_011447", "score.json")
N_PREDICT, D_MAX, PORT = 128, 4, 8090
ARGS = ["-m", os.path.join(CACHE, TARGET[0]), "-md", os.path.join(CACHE, DRAFT[0]), "--spec-type", "draft-simple",
        "--spec-draft-n-max", str(D_MAX), "--spec-draft-n-min", "0", "--spec-draft-p-min", "0",
        "-c", "4096", "-np", "1", "-t", "4", "-ngl", "0", "-ngld", "0", "--no-webui",
        "--host", "127.0.0.1", "--port", str(PORT)]
CHAT = "<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"   # Qwen3, thinking off
PROMPTS = [  # (id, category, language, text) -- fixed before any data
    ("e01", "explain", "en", "Explain how a refrigerator keeps food cold."),
    ("e02", "explain", "en", "What causes the seasons on Earth?"),
    ("e03", "explain", "en", "Why is the sky blue?"),
    ("e04", "explain", "en", "How does a vaccine train the immune system?"),
    ("e05", "explain", "en", "What is the difference between weather and climate?"),
    ("e06", "explain", "en", "Explain what inflation is to a high-school student."),
    ("h01", "howto", "en", "Give me five tips for studying for a final exam."),
    ("h02", "howto", "en", "How do I make a simple tomato pasta sauce?"),
    ("h03", "howto", "en", "How can I save money on groceries each month?"),
    ("h04", "howto", "en", "What should I check before buying a used bicycle?"),
    ("h05", "howto", "en", "How do I start running if I have never exercised before?"),
    ("h06", "howto", "en", "Suggest a weekly plan for learning basic Python."),
    ("c01", "code", "en", "Write a Python function that returns the n-th Fibonacci number iteratively."),
    ("c02", "code", "en", "Write a Python function that checks whether a string is a palindrome, ignoring case and spaces."),
    ("c03", "code", "en", "Explain what this Python expression does: [x * x for x in range(10) if x % 2 == 0]"),
    ("c04", "code", "en", "Write a SQL query that counts the orders of each customer in a table orders(customer_id, order_id)."),
    ("c05", "code", "en", "Write a bash command that counts the lines of every .txt file in a folder."),
    ("c06", "code", "en", "What is the difference between a list and a tuple in Python?"),
    ("c07", "code", "en", "Write a JavaScript function that debounces another function."),
    ("c08", "code", "en", "Explain Big-O notation with an example."),
    ("m01", "math", "en", "A train travels 180 km in 2.5 hours. What is its average speed? Show the steps."),
    ("m02", "math", "en", "A shirt costs 400 baht after a 20% discount. What was the original price?"),
    ("m03", "math", "en", "Solve 3x + 7 = 22 and explain each step."),
    ("m04", "math", "en", "What is the probability of rolling a sum of 7 with two dice?"),
    ("m05", "math", "en", "Convert 98.6 degrees Fahrenheit to Celsius, showing the formula."),
    ("m06", "math", "en", "A rectangle has a perimeter of 30 cm and a length of 9 cm. What is its area?"),
    ("w01", "writing", "en", "Write a short, polite email asking my professor for a meeting next week."),
    ("w02", "writing", "en", "Write a four-line poem about rain in the city."),
    ("w03", "writing", "en", "Summarize the plot of Romeo and Juliet in three sentences."),
    ("w04", "writing", "en", "Write a product description for a reusable water bottle."),
    ("w05", "writing", "en", "Write a short thank-you note to a friend who helped me move house."),
    ("w06", "writing", "en", "Rewrite this sentence to sound more formal: 'Hey, can you send me the file asap?'"),
    ("t01", "technical", "en", "What is a Raspberry Pi, and what can it be used for?"),
    ("t02", "technical", "en", "Explain what a large language model is in simple terms."),
    ("t03", "technical", "en", "What does a GPU do differently from a CPU?"),
    ("t04", "technical", "en", "Explain the difference between RAM and storage."),
    ("t05", "technical", "en", "What is an IP address?"),
    ("t06", "technical", "en", "What is the difference between TCP and UDP?"),
    ("t07", "technical", "en", "Why do computers slow down when they overheat?"),
    ("t08", "technical", "en", "What is the difference between Wi-Fi and Ethernet?"),
    ("th01", "explain", "th", "อธิบายว่าทำไมท้องฟ้าจึงเป็นสีฟ้า"),
    ("th02", "howto", "th", "แนะนำวิธีเตรียมตัวสอบปลายภาคให้มีประสิทธิภาพ 5 ข้อ"),
    ("th03", "writing", "th", "เขียนอีเมลสั้น ๆ ถึงอาจารย์เพื่อขอนัดพบในสัปดาห์หน้า"),
    ("th04", "technical", "th", "อธิบายความแตกต่างระหว่าง RAM กับหน่วยเก็บข้อมูล"),
    ("th05", "explain", "th", "สรุปประโยชน์ของการออกกำลังกายอย่างสม่ำเสมอ"),
    ("th06", "howto", "th", "วิธีทำไข่เจียวให้ฟูและกรอบ"),
    ("th07", "explain", "th", "อธิบายว่าปัญญาประดิษฐ์คืออะไร ให้นักเรียนประถมเข้าใจ"),
    ("th08", "code", "th", "เขียนฟังก์ชัน Python หาผลรวมของเลขคู่ในลิสต์ พร้อมอธิบาย"),
    ("th09", "math", "th", "ถ้าซื้อของ 3 ชิ้น ชิ้นละ 125 บาท และได้ส่วนลด 10% ต้องจ่ายเท่าไร"),
    ("th10", "howto", "th", "แนะนำสถานที่ท่องเที่ยวในเชียงใหม่ 3 แห่ง พร้อมเหตุผล"),
]
ALPHA1_BAR = 0.60


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def build():
    if len(PROMPTS) != 50 or len({p[0] for p in PROMPTS}) != 50:
        sys.exit("refused: 50 prompts with unique ids are required")
    s1 = json.load(open(STEP1))
    T = {k: v["mean_ms"] for k, v in s1["per_k"].items()}
    return {
        "registered": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
        "name": "spec_accept_20261010",
        "campaign": "scheduler upgrade 1, step 2: Qwen3-0.6B's acceptance as a draft for Qwen3-14B, on the laptop",
        "survey": "Documents/HeteroServe_Scheduler_Research.html, upgrade 1 (2026-10-10)",
        "target": {"file": TARGET[0], "sha256": TARGET[1]}, "draft": {"file": DRAFT[0], "sha256": DRAFT[1]},
        "server": SERVER, "server_args": ARGS, "chat_format": CHAT, "n_predict": N_PREDICT, "draft_n_max": D_MAX,
        "server_env": {"LLAMA_TRACE": "1"},   # the server logs each check ("accepted n/m draft tokens") only when traced
        "sampling": {"temperature": 0, "top_k": 1, "cache_prompt": False},
        "prompts": [{"id": i, "category": c, "lang": l, "text": t} for i, c, l, t in PROMPTS],
        "step1": {"score": os.path.relpath(STEP1, ROOT), "score_sha256": sha(STEP1), "T_ms": T},
        "predicted": {"alpha_each_position": 0.7, "alpha1_bar": ALPHA1_BAR},
        "inputs_sha256": {f: sha(os.path.join(HERE, f)) for f in (os.path.basename(__file__), "spec_accept_20261010.py")},
        "hypotheses": [{"id": "SA-PAY1", "claim": "alpha1 >= %.2f over every 4-token check of the 50 prompts" % ALPHA1_BAR}],
    }


def main(argv):
    if os.path.exists(OUT) and "--dry-run" not in argv:
        sys.exit("refused: %s exists; a registration is never overwritten" % os.path.relpath(OUT, ROOT))
    reg = build()
    print("%d prompts (%d Thai), n_predict %d, draft %d per check; step 1 T(k) %s" % (
        len(reg["prompts"]), sum(p["lang"] == "th" for p in reg["prompts"]), N_PREDICT, D_MAX, reg["step1"]["T_ms"]))
    print("server %s %s" % (SERVER, " ".join(ARGS)))
    if "--dry-run" in argv:
        print("dry run: nothing written"); return
    json.dump(reg, open(OUT, "w"), indent=1, ensure_ascii=False)
    os.chmod(OUT, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    print("registered %s at %s, sha256 %s (read-only)" % (os.path.relpath(OUT, ROOT), reg["registered"], sha(OUT)[:16]))


if __name__ == "__main__":
    main(sys.argv[1:])
