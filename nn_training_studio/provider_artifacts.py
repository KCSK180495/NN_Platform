"""Provider artifacts for NN Training Studio."""

import json
from nn_training_studio.ai_transport import (
    _provider_endpoint,
    _validate_provider_base_url,
)
from nn_training_studio.constants import (
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_OPENAI,
    APPLICATION_NAME,
    APP_VERSION,
    OFFICIAL_PROVIDER_SETTINGS,
)


def provider_api_compatibility_report(settings_owner, target, purpose):
    """Check whether the configured provider can be integrated safely."""
    failures = []
    warnings = []
    passed = []
    provider = settings_owner.ai_provider_var.get()
    model = settings_owner.ai_model_var.get().strip()
    base_url = settings_owner.ai_base_url_var.get().strip()
    if provider not in OFFICIAL_PROVIDER_SETTINGS:
        failures.append(
            "No online AI provider is selected. Configure OpenAI, DeepSeek, "
            "Claude, or an OpenAI-compatible provider in Settings."
        )
    else:
        passed.append(f"Provider selected: {provider}.")
    if not model:
        failures.append("No provider model name is configured.")
    else:
        passed.append(f"Provider model configured: {model}.")
    try:
        validated_url = _validate_provider_base_url(base_url)
        passed.append(f"Provider endpoint is valid: {validated_url}.")
    except Exception as exc:
        failures.append(str(exc))
    key_value = (
        settings_owner.resolve_ai_api_key(provider)
        if provider in OFFICIAL_PROVIDER_SETTINGS
        else ""
    )
    if not key_value:
        failures.append(
            "No API key is available for the selected provider. The generated "
            "application expects the key in an environment variable."
        )
    else:
        passed.append("A provider credential is available for testing.")
    connection_state = getattr(settings_owner, "ai_connection_state", "configured")
    if connection_state == "connected":
        passed.append("The provider connection was verified in this session.")
    else:
        warnings.append(
            "The provider is not visually marked Connected. Test it in Settings "
            "before relying on generated integration code."
        )
    if target in {"Python Application", "Windows Desktop / EXE"}:
        warnings.append(
            "Do not distribute a desktop program with a shared provider key. "
            "Use each user's key or call a protected backend owned by the application."
        )
    if target in {"Sensor / Serial Device", "Modbus TCP / PLC", "OPC UA"}:
        warnings.append(
            "A remote language-model API is not suitable for deterministic hard "
            "real-time or safety-critical actuation. Keep safety control local."
        )
    if "Vision" in purpose and target in {
        "Sensor / Serial Device", "Modbus TCP / PLC", "OPC UA"
    }:
        failures.append(
            "The selected hardware transport does not directly carry image data. "
            "Use a gateway or camera-capable application layer."
        )
    status = "Ready" if not failures else "Action required"
    if not failures and warnings:
        status = "Ready with warnings"
    return {
        "status": status,
        "failures": failures,
        "warnings": warnings,
        "passed": passed,
    }


def generate_ai_provider_api_artifacts(
    provider,
    model,
    base_url,
    purpose,
    target,
    protocol,
):
    """Generate key-free provider API integration files for an application."""
    if provider not in OFFICIAL_PROVIDER_SETTINGS:
        raise ValueError("Select an online AI provider before generating code.")
    base_url = _validate_provider_base_url(base_url)
    env_name = OFFICIAL_PROVIDER_SETTINGS[provider]["api_key_environment"]
    if provider == AI_PROVIDER_OPENAI:
        endpoint = _provider_endpoint(base_url, "/v1/responses")
        api_kind = "openai_responses"
    elif provider == AI_PROVIDER_ANTHROPIC:
        endpoint = _provider_endpoint(base_url, "/v1/messages")
        api_kind = "anthropic_messages"
    else:
        endpoint = _provider_endpoint(base_url, "/chat/completions")
        api_kind = "chat_completions"

    config = {
        "schema_version": 1,
        "created_by": f"{APPLICATION_NAME} {APP_VERSION}",
        "provider": provider,
        "model": model,
        "endpoint": endpoint,
        "api_kind": api_kind,
        "api_key_environment": env_name,
        "purpose": purpose,
        "deployment_target": target,
        "communication_protocol": protocol,
        "timeout_seconds": 60,
    }
    client_code = '''"""Generated provider client. The API key is read from the environment."""
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "ai_api_config.json").read_text(encoding="utf-8"))


def _extract_text(payload):
    kind = CONFIG["api_kind"]
    if kind == "openai_responses":
        if isinstance(payload.get("output_text"), str):
            return payload["output_text"]
        parts = []
        for output in payload.get("output", []):
            for item in output.get("content", []):
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\\n".join(parts)
    if kind == "anthropic_messages":
        return "\\n".join(
            item.get("text", "") for item in payload.get("content", [])
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return payload["choices"][0]["message"]["content"]


def ask_ai(user_text, system_text="You are a helpful application assistant."):
    """Send one application request and return plain response text."""
    api_key = os.environ.get(CONFIG["api_key_environment"], "").strip()
    if not api_key:
        raise RuntimeError(
            f"Set {CONFIG['api_key_environment']} before starting the application."
        )
    kind = CONFIG["api_kind"]
    if kind == "openai_responses":
        body = {
            "model": CONFIG["model"],
            "input": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "store": False,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
    elif kind == "anthropic_messages":
        body = {
            "model": CONFIG["model"],
            "system": system_text,
            "messages": [{"role": "user", "content": user_text}],
            "max_tokens": 2048,
        }
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        }
    else:
        body = {
            "model": CONFIG["model"],
            "messages": [
                {"role": "system", "content": system_text},
                {"role": "user", "content": user_text},
            ],
            "stream": False,
        }
        headers = {"Authorization": f"Bearer {api_key}"}
    headers["Content-Type"] = "application/json"
    request = urllib.request.Request(
        CONFIG["endpoint"],
        data=json.dumps(body).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urllib.request.urlopen(
            request,
            timeout=float(CONFIG.get("timeout_seconds", 60)),
        ) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:1000]
        raise RuntimeError(f"Provider request failed ({exc.code}): {detail}") from exc
    text = _extract_text(payload).strip()
    if not text:
        raise RuntimeError("The provider returned no response text.")
    return text


if __name__ == "__main__":
    print(ask_ai("Reply with: connection test passed"))
'''
    files = {
        "ai_provider_client.py": client_code,
        "ai_api_config.json": json.dumps(config, indent=2),
        "requirements.txt": "",
        "README.md": f'''# AI Provider API Integration

Provider: {provider}

Model: {model}

Purpose: {purpose}

Target: {target}

## Safe setup

1. Review `ai_api_config.json` and the complete source code.
2. Store the API key in the `{env_name}` environment variable.
3. Run `python ai_provider_client.py` for a private connection test.
4. Import `ask_ai` from `ai_provider_client.py` at the application point where
   the user explicitly requests the AI feature.
5. Validate provider output before using it in the application.

Never place the key in browser JavaScript, a public repository, logs, or an
installer. For a website, call the provider from a protected backend.
''',
    }
    if target in {"Local REST API", "Docker REST API", "Cloud API"}:
        files["server.py"] = '''"""Backend API wrapper for the generated provider client."""
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ai_provider_client import ask_ai

api = FastAPI(title="Application AI Integration")


class AIRequest(BaseModel):
    message: str = Field(min_length=1, max_length=20000)


@api.get("/health")
def health():
    return {"status": "ready"}


@api.post("/api/ai")
def run_ai(request: AIRequest):
    return {"result": ask_ai(request.message)}
'''
        files["web_client.js"] = '''export async function askApplicationAI(message) {
  const response = await fetch("/api/ai", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({message}),
  });
  if (!response.ok) throw new Error(`AI request failed: ${response.status}`);
  return (await response.json()).result;
}
'''
        files["requirements.txt"] = "fastapi>=0.110\nuvicorn>=0.29\n"
    if target == "Docker REST API":
        files["Dockerfile"] = '''FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000
CMD ["uvicorn", "server:api", "--host", "0.0.0.0", "--port", "8000"]
'''
        files[".dockerignore"] = ".env\n__pycache__\n*.pyc\n"
    return files
