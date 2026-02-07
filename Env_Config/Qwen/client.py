import os
import re
import json
import base64
import requests

QWEN_API_KEY = os.path.join(os.path.abspath(os.path.dirname(__file__)), ".qwen_api_key")

def get_key ():
    if os.path.exists (QWEN_API_KEY):
        return open (QWEN_API_KEY).read ().strip ()
    key = input ("API Key: ").strip ()
    open (QWEN_API_KEY, "w").write (key)
    os.chmod (QWEN_API_KEY, 0o600)
    return key


def img_to_data_url (path):
    with open (path, "rb") as f:
        b64 = base64.b64encode (f.read ()).decode ()
    return f"data:image/jpeg;base64,{b64}"

class QwenClient:
    def __init__ (self):
        self.key = get_key ()
        self.url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
        self.model = "qwen-vl-max-latest"
        self.history = []

    def ask (self, text, img_paths,
        temperature = 0.0,
        max_tokens = 16384,
    ):
        message = {
            "role": "user",
            "content": [
                { "type": "text", "text": text },
            ]
        }
        for i, img_path in enumerate (img_paths):
            message["content"].append (
                { "type": "image_url", "image_url": {"url": img_to_data_url (img_path)} }
            )
        self.history.append(message)

        payload = {
            "model": self.model,
            "messages": self.history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(15, 600),
        )
        resp.raise_for_status()
        data = resp.json()

        def _pick_text(d):
            ch0 = (d.get("choices") or [None])[0] or {}
            msg = ch0.get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    return c.strip()
                if isinstance(c, list):
                    out = []
                    for p in c:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content")
                            if isinstance(t, str) and p.get("type") not in ("reasoning", "reasoning_content"):
                                out.append(t)
                    if out:
                        return "".join(out).strip()

            for k in ("text", "content", "output_text"):
                v = ch0.get(k)
                if isinstance(v, str):
                    return v.strip()
                if isinstance(v, list):
                    out = []
                    for p in v:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content")
                            if isinstance(t, str) and p.get("type") not in ("reasoning", "reasoning_content"):
                                out.append(t)
                    if out:
                        return "".join(out).strip()

            for k in ("output_text", "content", "text", "answer"):
                v = d.get(k)
                if isinstance(v, str):
                    return v.strip()
            return ""

        full_ans = _pick_text(data)
        print (f"[full_ans] = {full_ans[ : 20]}")
        # self.history.append({"role": "assistant", "content": full_ans})

        matches = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", full_ans)
        if not matches:
            return None
        x_str, y_str = matches[-1]
        return full_ans, int(x_str), int(y_str)
    
    def ask_4stir (self, text, img_paths,
        temperature = 0.0,
        max_tokens = 16384,
    ):
        message = {
            "role": "user",
            "content": [
                { "type": "text", "text": text },
            ]
        }
        for i, img_path in enumerate (img_paths):
            message["content"].append (
                { "type": "image_url", "image_url": {"url": img_to_data_url (img_path)} }
            )
        self.history.append(message)

        payload = {
            "model": self.model,
            "messages": self.history,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        resp = requests.post(
            self.url,
            headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(15, 600),
        )
        resp.raise_for_status()
        data = resp.json()

        def _pick_text(d):
            ch0 = (d.get("choices") or [None])[0] or {}
            msg = ch0.get("message")
            if isinstance(msg, dict):
                c = msg.get("content")
                if isinstance(c, str):
                    return c.strip()
                if isinstance(c, list):
                    out = []
                    for p in c:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content")
                            if isinstance(t, str) and p.get("type") not in ("reasoning", "reasoning_content"):
                                out.append(t)
                    if out:
                        return "".join(out).strip()

            for k in ("text", "content", "output_text"):
                v = ch0.get(k)
                if isinstance(v, str):
                    return v.strip()
                if isinstance(v, list):
                    out = []
                    for p in v:
                        if isinstance(p, dict):
                            t = p.get("text") or p.get("content")
                            if isinstance(t, str) and p.get("type") not in ("reasoning", "reasoning_content"):
                                out.append(t)
                    if out:
                        return "".join(out).strip()

            for k in ("output_text", "content", "text", "answer"):
                v = d.get(k)
                if isinstance(v, str):
                    return v.strip()
            return ""

        full_ans = _pick_text(data)
        print (f"[full_ans] = {full_ans[ : 20]}")
        # self.history.append({"role": "assistant", "content": full_ans})
        
        return full_ans

    def clear_memory (self):
        self.history.clear ()
