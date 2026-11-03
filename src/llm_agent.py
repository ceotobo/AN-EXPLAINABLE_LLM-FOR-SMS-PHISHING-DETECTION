# llm_agent.py -- LLM-based smishing detection agent via Ollama
# Optimized for Google Colab L4 GPU (24GB VRAM)
# Models    : Mistral-7B (mistral) and Llama 3-8B (llama3)
# Configs   : config1, config2 (classifier), config2c (explainer)
# Dataset   : test.csv (default) or outputs/shap_sample.csv (--shap-aligned)
#
# Run with  : python llm_agent.py --shap-aligned --configs config1,config2
#             python llm_agent.py --shap-aligned --configs config2c

import json, re, time, os, requests, sys, subprocess, argparse, glob
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.model_selection import train_test_split

# ── Command-line flags ────────────────────────────────────────────────────────
_parser = argparse.ArgumentParser()
_parser.add_argument("--shap-aligned", action="store_true",
                     help="Run on the 100-message SHAP sample instead of test.csv")
_parser.add_argument("--configs", type=str, default=None,
                     help="Comma-separated list of configs, e.g. 'config1,config2'")
_args = _parser.parse_args()

# ── Configuration ─────────────────────────────────────────────────────────────
OLLAMA_URL      = "http://localhost:11434/api/generate"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"
MODELS          = ["mistral", "llama3"]

if _args.shap_aligned:
    TEST_CSV    = "outputs/shap_sample.csv"
    OUTPUT_DIR  = "outputs"
    CHECKPOINT  = os.path.join(OUTPUT_DIR, "checkpoint_shap100.json")
    _INPUT_TAG  = "_100"
else:
    TEST_CSV    = "test.csv"
    OUTPUT_DIR  = "outputs"
    CHECKPOINT  = os.path.join(OUTPUT_DIR, "checkpoint.json")
    _INPUT_TAG  = ""

os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_SIZE = None

# ── Prompt definitions ────────────────────────────────────────────────────────
CONFIG_1_PROMPT = """You are a cybersecurity classifier specialising in SMS messages.

A PHISHING message tries to steal credentials, money, or personal information by impersonating banks, government agencies, or delivery services. It typically contains urgent threats, suspicious links, or requests to click/call/verify.

A LEGITIMATE message is a normal everyday text — a conversation, a genuine bank alert, an appointment reminder, or a real delivery notification with no suspicious request.

Classify the following SMS message.

SMS: {message}

Respond with exactly one word: PHISHING or LEGITIMATE."""


CONFIG_2_PROMPT = """You are a cybersecurity analyst specialising in SMS phishing detection.

PHISHING: Messages that impersonate trusted organisations to steal credentials, money, or personal data. They use urgency, threats, suspicious URLs, or requests to call premium numbers.

LEGITIMATE: Normal everyday SMS messages — conversations, genuine alerts, appointment reminders, delivery notifications with no malicious intent.

Analyse the following SMS and classify it.

SMS: {message}

Respond in exactly this format:
VERDICT: [PHISHING or LEGITIMATE]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
EXPLANATION: [One to three sentences explaining your decision]"""


CONFIG_2C_PROMPT = """You have classified the following SMS as {verdict}.

SMS: {message}

Now explain why it is {verdict}. Complete the steps below.

STEP 1 - EVIDENCE:
What specific words or phrases in the message support the verdict?

STEP 2 - CONTEXT:
Does the message contain links, phone numbers, or requests that are consistent with the verdict?

STEP 3 - CITED WORDS:
List the 3-5 most important tokens that justify the verdict, in quotes.

EXPLANATION:
Write one or two sentences a general user would understand."""


# ── Config options (generation limits) ───────────────────────────────────────
CONFIG_OPTIONS = {
    "config1":  {"num_predict": 32,  "timeout": 120,
                 "stop": ["\n\n", "SMS:", "ANOTHER SMS:", "NEXT MESSAGE:"]},
    "config2":  {"num_predict": 160, "timeout": 120,
                 "stop": ["\n\n", "SMS:", "ANOTHER SMS:", "NEXT MESSAGE:"]},
    "config2c": {"num_predict": 300, "timeout": 150,
                 "stop": ["SMS:", "ANOTHER SMS:", "NEXT MESSAGE:"]},
}

WARMUP_TIMEOUT   = 600
WARMUP_KEEPALIVE = "30m"
MAX_CONSECUTIVE_FAILURES = 3
BATCH_SAVE_INTERVAL = 5

BAD_MESSAGES = set()


# ── Ollama helpers ───────────────────────────────────────────────────────────
def ensure_ollama_running():
    print("  [Ollama] Checking if Ollama is running...")
    for attempt in range(3):
        try:
            r = requests.get(OLLAMA_TAGS_URL, timeout=10)
            if r.status_code == 200:
                print("  [Ollama] Ollama is running and responsive.")
                return True
        except:
            pass
        print(f"  [Ollama] Ollama not responding (attempt {attempt+1}/3)")
        print("  [Ollama] Restarting Ollama...")
        os.system("pkill -f ollama || true")
        time.sleep(2)
        os.system("nohup ollama serve > /tmp/ollama.log 2>&1 &")
        time.sleep(5)
        for wait_attempt in range(10):
            try:
                r = requests.get(OLLAMA_TAGS_URL, timeout=5)
                if r.status_code == 200:
                    print("  [Ollama] Ollama restarted successfully.")
                    return True
            except:
                pass
            time.sleep(2)
    print("  [Ollama] ERROR: Could not start Ollama.")
    return False


def pull_model_if_missing(model):
    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=10)
        available = [m['name'] for m in r.json().get('models', [])]
        base = model.split(':')[0]
        if any(base in a for a in available):
            print(f"  [Model] {model} is available.")
            return True
        print(f"  [Model] {model} not found. Pulling now...")
        process = subprocess.Popen(
            ['ollama', 'pull', model],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1)
        for line in process.stdout:
            if 'downloading' in line.lower() or 'pulling' in line.lower():
                print(f"    {line.strip()}")
            elif 'verifying' in line.lower():
                print(f"    {line.strip()}")
        process.wait()
        if process.returncode == 0:
            print(f"  [Model] Successfully pulled {model}")
            return True
        print(f"  [Model] Failed to pull {model}")
        return False
    except Exception as e:
        print(f"  [Model] Error checking/pulling model: {e}")
        return False


def warmup_model(model):
    print(f"  [Warmup] Loading {model} ...")
    t0 = time.time()
    try:
        requests.get(OLLAMA_TAGS_URL, timeout=5)
    except:
        print("  [Warmup] Ollama not responding. Restarting...")
        if not ensure_ollama_running():
            return False
    if not pull_model_if_missing(model):
        return False
    payload = {
        "model": model,
        "prompt": "Reply with exactly one word: OK",
        "stream": False,
        "keep_alive": WARMUP_KEEPALIVE,
        "options": {"num_predict": 5, "num_ctx": 512}
    }
    for attempt in range(3):
        try:
            print(f"  [Warmup] Attempt {attempt+1}/3...")
            r = requests.post(OLLAMA_URL, json=payload, timeout=WARMUP_TIMEOUT)
            r.raise_for_status()
            elapsed = time.time() - t0
            print(f"  [Warmup] {model} responded in {elapsed:.1f}s.")
            os.system("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader")
            os.system("ollama ps")
            return True
        except requests.exceptions.Timeout:
            print(f"  [Warmup] TIMEOUT after {WARMUP_TIMEOUT}s (attempt {attempt+1}/3)")
            if attempt < 2:
                try:
                    requests.post(OLLAMA_URL, json={"model": model, "prompt": "", "stream": False, "keep_alive": 0}, timeout=10)
                except:
                    pass
                time.sleep(5)
                if attempt == 1:
                    ensure_ollama_running()
        except Exception as e:
            print(f"  [Warmup] FAILED for {model}: {e}")
            if attempt < 2:
                time.sleep(5)
    print(f"  [Warmup] FATAL: Could not load {model} after 3 attempts.")
    return False


def reset_ollama_state(model):
    print("  [Reset] Attempting to reset Ollama state...")
    try:
        requests.post(OLLAMA_URL, json={"model": model, "prompt": "", "stream": False, "keep_alive": 0}, timeout=10)
        time.sleep(3)
        import gc
        gc.collect()
        success = warmup_model(model)
        print("  [Reset] Successfully reset" if success else "  [Reset] Failed to reload")
        return success
    except Exception as e:
        print(f"  [Reset] Failed: {e}")
        return False


def unload_model(model):
    print(f"  [Unload] Removing {model} from Ollama memory ...")
    try:
        r = requests.post(OLLAMA_URL, json={"model": model, "prompt": "", "stream": False, "keep_alive": 0}, timeout=30)
        r.raise_for_status()
        time.sleep(3)
        import gc
        gc.collect()
        os.system("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader")
        os.system("ollama ps")
        return True
    except Exception as e:
        print(f"  [Unload] Warning: {e}")
        return False


def call_ollama(model, prompt, timeout=120, max_retries=4, num_predict=400, stop_sequences=None):
    if stop_sequences is None:
        stop_sequences = ["\n\n", "SMS:", "ANOTHER SMS:", "NEXT MESSAGE:"]
    payload = {
        "model"  : model,
        "prompt" : prompt,
        "stream" : False,
        "keep_alive": WARMUP_KEEPALIVE,
        "options": {
            "temperature": 0.0,
            "num_predict": num_predict,
            "num_ctx"    : 2048,
            "top_p"      : 1.0,
            "top_k"      : 40,
            "seed"       : 42,
            "stop": stop_sequences,
        }
    }
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                wait_time = min(2 ** attempt, 30) + np.random.uniform(0, 1)
                print(f"    Retry {attempt}/{max_retries}, waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            current_timeout = timeout + (attempt * 30)
            response = requests.post(OLLAMA_URL, json=payload, timeout=current_timeout)
            response.raise_for_status()
            text = response.json().get("response", "").strip()
            if len(text) > 3000:
                print(f"    WARNING: Response too long, truncating.")
                text = text[:3000]
            if not text:
                print(f"    Empty response (attempt {attempt+1}/{max_retries+1})")
                continue
            return text
        except requests.exceptions.Timeout:
            print(f"    Timeout (attempt {attempt+1}/{max_retries+1})")
            try:
                requests.post(OLLAMA_URL, json={"model": model, "prompt": "OK", "stream": False,
                                                "keep_alive": "1m", "options": {"num_predict": 2}}, timeout=10)
            except Exception:
                pass
        except requests.exceptions.ConnectionError:
            print(f"    Connection lost (attempt {attempt+1}/{max_retries+1})")
            time.sleep(5)
        except Exception as e:
            print(f"    Error: {e}")
            time.sleep(2)
    return None


# ── Parsers ──────────────────────────────────────────────────────────────────
def parse_config1(response):
    if response is None:
        return {"verdict": None, "confidence": None, "cot_trace": None, "cited_tokens": [],
                "tech_explanation": None, "user_explanation": None, "parse_error": "null_response"}
    text = response.strip().upper()
    for word in text.split():
        clean = re.sub(r'[^A-Z]', '', word)
        if clean in ["PHISHING", "LEGITIMATE"]:
            return {"verdict": clean, "confidence": None, "cot_trace": None, "cited_tokens": [],
                    "tech_explanation": None, "user_explanation": None, "parse_error": None}
    return {"verdict": None, "confidence": None, "cot_trace": None, "cited_tokens": [],
            "tech_explanation": None, "user_explanation": None,
            "parse_error": f"no_verdict_found: {response[:80]}"}


def parse_config2(response):
    if response is None:
        return {"verdict": None, "confidence": None, "cot_trace": None, "cited_tokens": [],
                "tech_explanation": None, "user_explanation": None, "parse_error": "null_response"}
    result = {"verdict": None, "confidence": None, "cot_trace": None, "cited_tokens": [],
              "tech_explanation": None, "user_explanation": None, "parse_error": None}
    m = re.search(r'VERDICT\s*:\s*(PHISHING|LEGITIMATE)', response, re.IGNORECASE)
    if m:
        result["verdict"] = m.group(1).upper()
    else:
        result["parse_error"] = f"no_verdict: {response[:80]}"
    m = re.search(r'CONFIDENCE\s*:\s*(HIGH|MEDIUM|LOW)', response, re.IGNORECASE)
    if m:
        result["confidence"] = m.group(1).upper()
    m = re.search(r'(?:EXPLANATION|REASON)\s*:\s*(.+)', response, re.IGNORECASE | re.DOTALL)
    if m:
        result["tech_explanation"] = m.group(1).strip()
    return result


def parse_config2c(response):
    """Parser for config2c: extracts STEP 3 CITED WORDS and EXPLANATION.

    Only tokens inside quotes are accepted. The main loop further filters
    against the source message.
    """
    if response is None:
        return {"verdict": None, "confidence": None, "cot_trace": None, "cited_tokens": [],
                "tech_explanation": None, "user_explanation": None, "parse_error": "null_response"}

    result = {"verdict": None, "confidence": None, "cot_trace": response,
              "cited_tokens": [], "tech_explanation": None,
              "user_explanation": None, "parse_error": None}

    cited_patterns = [
        r'STEP\s*3[^:]*:\s*(.*?)(?=EXPLANATION|\Z)',
        r'CITED WORDS\s*:?\s*(.*?)(?=EXPLANATION|\Z)',
    ]
    quoted_tokens = []
    for pattern in cited_patterns:
        m = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1)
            quoted = re.findall(
                r'["\u2018\u2019\u201c\u201d]([^"\']{2,40})["\u2018\u2019\u201c\u201d]',
                text)
            for t in quoted:
                for w in t.split():
                    w = w.strip().lower()
                    if len(w) > 1:
                        quoted_tokens.append(w)
            break

    seen = set()
    cited = []
    for t in quoted_tokens:
        if t not in seen:
            seen.add(t)
            cited.append(t)

    result["cited_tokens"] = cited[:10]

    m = re.search(r'EXPLANATION\s*:?\s*(.*?)$', response, re.IGNORECASE | re.DOTALL)
    if m:
        result["tech_explanation"] = m.group(1).strip()
        result["user_explanation"] = result["tech_explanation"]
    return result


# ── Config registry ──────────────────────────────────────────────────────────
CONFIGS = {
    "config1":  {"prompt": CONFIG_1_PROMPT,  "parser": parse_config1,
                 "description": "Minimal classification prompt"},
    "config2":  {"prompt": CONFIG_2_PROMPT,  "parser": parse_config2,
                 "description": "Verdict + short explanation (primary classifier)"},
    "config2c": {"prompt": CONFIG_2C_PROMPT, "parser": parse_config2c,
                 "description": "Post-hoc CoT explanation of a fixed verdict"},
}

# ── Filter configs ───────────────────────────────────────────────────────────
if _args.configs:
    _requested = [c.strip() for c in _args.configs.split(",") if c.strip()]
    invalid = [c for c in _requested if c not in CONFIGS]
    if invalid:
        print(f"ERROR: unknown configs {invalid}. Valid: {list(CONFIGS.keys())}")
        sys.exit(1)
    CONFIGS_TO_RUN = {k: CONFIGS[k] for k in _requested}
    _CONFIG_TAG = "_" + "_".join(_requested) if _requested != list(CONFIGS.keys()) else ""
else:
    CONFIGS_TO_RUN = CONFIGS
    _CONFIG_TAG = ""

_OUTPUT_TAG = _INPUT_TAG + _CONFIG_TAG


# ── Helpers ──────────────────────────────────────────────────────────────────
def verdict_to_label(verdict):
    if verdict == "PHISHING":
        return 1
    if verdict == "LEGITIMATE":
        return 0
    return None


def compute_metrics(y_true, y_pred, model_name, config_name):
    pairs = [(yt, yp) for yt, yp in zip(y_true, y_pred) if yp is not None]
    fail_count = sum(1 for yp in y_pred if yp is None)
    if not pairs:
        return {"model": model_name, "config": config_name,
                "accuracy": None, "precision": None, "recall": None,
                "f1": None, "fpr": None, "n_evaluated": 0,
                "n_parse_failures": fail_count}
    yt_clean = [p[0] for p in pairs]
    yp_clean = [p[1] for p in pairs]
    acc  = accuracy_score(yt_clean, yp_clean)
    prec = precision_score(yt_clean, yp_clean, average='binary', zero_division=0)
    rec  = recall_score(yt_clean, yp_clean, average='binary', zero_division=0)
    f1v  = f1_score(yt_clean, yp_clean, average='binary', zero_division=0)
    tn, fp, fn, tp = confusion_matrix(yt_clean, yp_clean, labels=[0,1]).ravel()
    fpr  = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"model": model_name, "config": config_name,
            "accuracy": round(acc,4), "precision": round(prec,4),
            "recall": round(rec,4), "f1": round(f1v,4), "fpr": round(fpr,4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "n_evaluated": len(pairs), "n_parse_failures": fail_count}


def get_processed_count(all_results, model, config_name):
    return len([r for r in all_results
                if r.get('model') == model and r.get('config') == config_name])


def is_config_complete(all_results, model, config_name, total_messages):
    return get_processed_count(all_results, model, config_name) >= total_messages


def save_checkpoint(progress, all_results, all_metrics, cited_tokens):
    ckpt = {"progress": progress, "results": all_results, "metrics": all_metrics,
            "cited_tokens": cited_tokens, "bad_messages": list(BAD_MESSAGES),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(CHECKPOINT, 'w') as f:
        json.dump(ckpt, f, indent=2)


def load_checkpoint():
    if not os.path.exists(CHECKPOINT):
        return {}, [], [], {}, []
    with open(CHECKPOINT) as f:
        ckpt = json.load(f)
    global BAD_MESSAGES
    BAD_MESSAGES = set(ckpt.get("bad_messages", []))
    print(f"  [Resume] Loaded checkpoint from {ckpt.get('timestamp', 'unknown')}")
    return (ckpt.get("progress", {}), ckpt.get("results", []),
            ckpt.get("metrics", []), ckpt.get("cited_tokens", []),
            list(BAD_MESSAGES))


def load_config2_verdicts(model):
    """Load verdicts from config2 (the working classifier) for use by config2c.

    Searches all llm_agent_results_100*.json files for config2 entries,
    because the filename varies depending on the --configs argument used.
    """
    pattern = f"{OUTPUT_DIR}/llm_agent_results{_INPUT_TAG}*.json"
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        print(f"  WARNING: no results files found matching {pattern}")
        return {}
    lookup = {}
    for path in candidates:
        try:
            with open(path) as f:
                data = json.load(f)
        except Exception:
            continue
        for r in data:
            if (r.get('config') == 'config2'
                    and r.get('model') == model
                    and r.get('verdict')):
                lookup[r['idx']] = r['verdict']
    print(f"  [config2c] Loaded {len(lookup)} verdicts from config2 for {model} "
          f"(from {len(candidates)} file(s))")
    return lookup


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 80)
    print(f"LLM SMISHING DETECTION  (input={_INPUT_TAG or 'full'}  "
          f"configs={list(CONFIGS_TO_RUN.keys())})")
    print("=" * 80)

    if not ensure_ollama_running():
        print("ERROR: Cannot proceed without Ollama.")
        return

    try:
        r = requests.get(OLLAMA_TAGS_URL, timeout=10)
        available = [m['name'] for m in r.json().get('models', [])]
        print(f"Available models: {available}")
    except:
        available = []

    print(f"\nLoading test set from {TEST_CSV} ...")
    if not os.path.exists(TEST_CSV):
        print(f"  ERROR: {TEST_CSV} not found!")
        return

    test_df = pd.read_csv(TEST_CSV).sample(frac=1, random_state=42).reset_index(drop=True)
    if SAMPLE_SIZE is not None and SAMPLE_SIZE < len(test_df):
        test_df, _ = train_test_split(test_df, train_size=SAMPLE_SIZE,
                                      stratify=test_df['label_binary'], random_state=42)
        print(f"Using stratified sample of {SAMPLE_SIZE} messages.")

    total_messages = len(test_df)
    print(f"Test messages: {total_messages} "
          f"(ham={(test_df['label_binary']==0).sum()}, "
          f"malicious={(test_df['label_binary']==1).sum()})")

    os.system("nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv,noheader")

    resume_from, all_results, all_metrics, cited_tokens, _ = load_checkpoint()
    if resume_from:
        print(f"\n  [Resume] Progress: {resume_from}")

    for model_idx, model in enumerate(MODELS):
        print("\n" + "=" * 80)
        print(f"MODEL {model_idx+1}/{len(MODELS)}: {model}")
        print("=" * 80)

        base = model.split(':')[0]
        if not any(base in a for a in available):
            if not pull_model_if_missing(model):
                print(f"  Skipping {model}")
                continue
            try:
                r = requests.get(OLLAMA_TAGS_URL, timeout=10)
                available = [m['name'] for m in r.json().get('models', [])]
            except:
                pass

        if not warmup_model(model):
            print(f"  FATAL: could not load {model}.")
            continue

        verdict_lookup = {}
        if 'config2c' in CONFIGS_TO_RUN:
            verdict_lookup = load_config2_verdicts(model)

        for config_idx, (config_name, config) in enumerate(CONFIGS_TO_RUN.items()):
            config_key = f"{model}_{config_name}"
            opts = CONFIG_OPTIONS[config_name]

            processed_count = get_processed_count(all_results, model, config_name)

            if is_config_complete(all_results, model, config_name, total_messages):
                print(f"\n{'='*60}")
                print(f"Model: {model} | Config: {config_name} -- ALREADY COMPLETE")
                print(f"{'='*60}")
                continue

            print(f"\n{'='*60}")
            print(f"Config {config_idx+1}/{len(CONFIGS_TO_RUN)}: {config_name}")
            print(f"  Description: {config['description']}")
            print(f"  Processed: {processed_count}/{total_messages}")
            print(f"{'='*60}")

            if config_key not in cited_tokens:
                cited_tokens[config_key] = {}

            y_true, y_pred = [], []
            config_results = []
            messages_done = 0
            config_start = time.time()
            consecutive_failures = 0
            last_checkpoint_save = 0

            for idx, row in test_df.iterrows():
                msg = str(row['text'])
                label = int(row['label_binary'])

                if messages_done < processed_count:
                    messages_done += 1
                    continue

                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    print(f"  [Health] resetting Ollama...")
                    reset_ollama_state(model)
                    consecutive_failures = 0
                    time.sleep(5)

                if msg in BAD_MESSAGES:
                    response = None
                else:
                    if config_name == 'config2c':
                        verdict = verdict_lookup.get(int(idx), 'PHISHING')
                        prompt = config['prompt'].format(message=msg, verdict=verdict)
                    else:
                        prompt = config['prompt'].format(message=msg)

                    response = call_ollama(model, prompt,
                                           timeout=opts["timeout"],
                                           num_predict=opts["num_predict"],
                                           stop_sequences=opts["stop"])

                if response is None and msg not in BAD_MESSAGES:
                    BAD_MESSAGES.add(msg)
                    consecutive_failures += 1
                else:
                    consecutive_failures = 0

                parsed = config['parser'](response)

                # For config2c, verdict is inherited from config2 (fixed)
                if config_name == 'config2c':
                    parsed['verdict'] = verdict_lookup.get(int(idx))
                    parsed['parse_error'] = None
                    # Keep only cited tokens that appear in the source message
                    msg_words = set(re.findall(r"[a-z0-9']+", msg.lower()))
                    parsed['cited_tokens'] = [
                        t for t in parsed['cited_tokens'] if t in msg_words
                    ]

                pred_label = verdict_to_label(parsed['verdict'])
                y_true.append(label)
                y_pred.append(pred_label)

                if config_name == 'config2c' and parsed['cited_tokens']:
                    cited_tokens[config_key][str(idx)] = parsed['cited_tokens']

                all_results.append({
                    "idx": int(idx), "model": model, "config": config_name,
                    "text": msg, "true_label": label, "pred_label": pred_label,
                    "verdict": parsed['verdict'], "confidence": parsed['confidence'],
                    "cot_trace": parsed['cot_trace'], "cited_tokens": parsed['cited_tokens'],
                    "tech_explanation": parsed['tech_explanation'],
                    "user_explanation": parsed['user_explanation'],
                    "parse_error": parsed['parse_error'], "raw_response": response,
                })
                config_results.append(all_results[-1])
                messages_done += 1

                if messages_done % 10 == 0:
                    correct = sum(1 for r in config_results
                                  if r['pred_label'] is not None
                                  and r['pred_label'] == r['true_label'])
                    evaluated = sum(1 for r in config_results
                                    if r['pred_label'] is not None)
                    running_acc = correct / evaluated if evaluated > 0 else 0
                    elapsed = time.time() - config_start
                    rate = messages_done / elapsed if elapsed > 0 else 0
                    eta = (total_messages - messages_done) / rate if rate > 0 else 0
                    print(f"  [{messages_done}/{total_messages}] Acc: {running_acc:.3f} "
                          f"| {rate:.2f} msg/s | ETA: {eta/60:.1f}min")

                if messages_done - last_checkpoint_save >= BATCH_SAVE_INTERVAL:
                    resume_from[config_key] = messages_done
                    save_checkpoint(resume_from, all_results, all_metrics, cited_tokens)
                    last_checkpoint_save = messages_done
                    with open(f"{OUTPUT_DIR}/llm_agent_results{_OUTPUT_TAG}.json", 'w') as f:
                        json.dump(all_results, f, indent=2)
                    with open(f"{OUTPUT_DIR}/llm_agent_cited_tokens{_OUTPUT_TAG}.json", 'w') as f:
                        json.dump(cited_tokens, f, indent=2)

            config_results_from_all = [r for r in all_results
                                       if r['model'] == model and r['config'] == config_name]
            y_true_ck = [r['true_label'] for r in config_results_from_all]
            y_pred_ck = [r['pred_label'] for r in config_results_from_all]
            metrics = compute_metrics(y_true_ck, y_pred_ck, model, config_name)

            existing_idx = next((i for i, m in enumerate(all_metrics)
                                 if m['model'] == model and m['config'] == config_name), None)
            if existing_idx is not None:
                all_metrics[existing_idx] = metrics
            else:
                all_metrics.append(metrics)

            print(f"\n  Results for {model} / {config_name} "
                  f"({(time.time()-config_start)/60:.1f} min):")
            print(f"    Accuracy  : {metrics['accuracy']}")
            print(f"    F1-Score  : {metrics['f1']}")
            print(f"    FPR       : {metrics['fpr']}")

            resume_from[config_key] = total_messages
            save_checkpoint(resume_from, all_results, all_metrics, cited_tokens)
            with open(f"{OUTPUT_DIR}/llm_agent_results{_OUTPUT_TAG}.json", 'w') as f:
                json.dump(all_results, f, indent=2)
            pd.DataFrame(all_metrics).to_csv(
                f"{OUTPUT_DIR}/llm_agent_metrics{_OUTPUT_TAG}.csv", index=False)
            with open(f"{OUTPUT_DIR}/llm_agent_cited_tokens{_OUTPUT_TAG}.json", 'w') as f:
                json.dump(cited_tokens, f, indent=2)

        unload_model(model)

    print(f"\n{'='*80}\nFINAL SUMMARY\n{'='*80}")
    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        print(metrics_df[['model','config','accuracy','precision','recall','f1','fpr']].to_string(index=False))
    print(f"\nAll outputs saved with tag: '{_OUTPUT_TAG or 'default'}'")


if __name__ == "__main__":
    main()