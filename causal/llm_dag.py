"""
causal/llm_dag.py

LLM-guided parameter initialization using Groq API.
Model: llama-3.3-70b-versatile (free tier at console.groq.com)
Uses curl subprocess — urllib has SSL issues on MSI.

Setup:
  echo "gsk_YOUR_KEY" > causal/.groq_key
  chmod 600 causal/.groq_key

If no key found, training falls back to random init near Hahn & Stock 2001.
"""

import os, json, subprocess, numpy as np

_KEY_FILE = os.path.join(os.path.dirname(__file__), '.groq_key')


def llm_initialize_params(train_datasets: dict) -> np.ndarray:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from physics.hamiltonian import PARAM_NAMES, HAHN_STOCK_2001, params_to_vec

    if not os.path.exists(_KEY_FILE):
        raise RuntimeError(f"No key file at {_KEY_FILE}")
    key = open(_KEY_FILE).read().strip()

    prompt = (
        "Return ONLY a JSON object with 8 KDC Hamiltonian parameters for pyrazine "
        "S2->S1 internal conversion. Vary each by a random amount up to 5% from "
        "the Hahn & Stock 2001 literature values. No explanation, just JSON.\n"
        "Literature: {\"E_S1\": 3.9950, \"E_S2\": 4.9183, \"om1\": 0.1273, "
        "\"om10a\": 0.1133, \"kap1_S1\": -0.0470, \"kap1_S2\": -0.2012, "
        "\"gamma\": -0.0180, \"lam\": 0.1825}"
    )

    payload = json.dumps({
        "model"      : "llama-3.3-70b-versatile",
        "messages"   : [{"role": "user", "content": prompt}],
        "max_tokens" : 200,
        "temperature": 0.4,
    })

    result = subprocess.run([
        "curl", "-s",
        "https://api.groq.com/openai/v1/chat/completions",
        "-H", f"Authorization: Bearer {key}",
        "-H", "Content-Type: application/json",
        "-d", payload,
    ], capture_output=True, text=True, timeout=20)

    content = json.loads(result.stdout)["choices"][0]["message"]["content"]
    start   = content.find("{"); end = content.rfind("}") + 1
    params  = json.loads(content[start:end])

    vec = params_to_vec(HAHN_STOCK_2001).copy()
    for i, name in enumerate(PARAM_NAMES):
        if name in params:
            vec[i] = float(params[name])

    print(f"  LLM initialization successful")
    for i, name in enumerate(PARAM_NAMES):
        print(f"    {name:12s}: {vec[i]:+.4f}  (lit: {params_to_vec(HAHN_STOCK_2001)[i]:+.4f})")
    return vec
