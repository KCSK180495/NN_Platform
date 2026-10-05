"""Ai transport for NN Training Studio."""

import json
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlparse
from nn_training_studio.ai_validation import (
    AIProviderError,
)
from nn_training_studio.constants import (
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_OPENAI,
    OFFICIAL_PROVIDER_SETTINGS,
)


def _validate_provider_base_url(base_url):
    value = str(base_url).strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme != "https" or not parsed.netloc:
        raise AIProviderError(
            "Provider Base URL must be a complete HTTPS URL."
        )
    if parsed.username or parsed.password:
        raise AIProviderError(
            "Provider Base URL must not contain embedded credentials."
        )
    if parsed.query or parsed.fragment:
        raise AIProviderError(
            "Provider Base URL must not contain query parameters or fragments."
        )
    return value


def _provider_endpoint(base_url, endpoint_path):
    base = base_url.rstrip("/")
    path = "/" + endpoint_path.lstrip("/")
    if path.startswith("/v1/") and base.endswith("/v1"):
        path = path[3:]
    return base + path


def _masked_api_key(api_key):
    value = str(api_key or "").strip()
    if not value:
        return "No key configured"
    suffix = value[-4:] if len(value) >= 4 else value
    return f"••••••••{suffix}"


def _read_json_response(request, timeout_seconds):
    try:
        with urllib_request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            response_text = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_error = json.loads(error_body)
            detail = (
                parsed_error.get("error", {}).get("message")
                or parsed_error.get("message")
                or error_body
            )
        except Exception:
            detail = error_body
        raise AIProviderError(
            f"Provider returned HTTP {exc.code}: {detail[:800]}"
        ) from exc
    except urllib_error.URLError as exc:
        raise AIProviderError(
            f"Could not connect to the AI provider: {exc.reason}"
        ) from exc

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise AIProviderError(
            "The provider returned a non-JSON HTTP response."
        ) from exc


def test_ai_provider_connection(
    provider_name,
    api_key,
    model,
    base_url,
    timeout_seconds=30,
    transport=None,
):
    """Validate authentication without sending a dataset or generation prompt."""
    if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
        raise AIProviderError(
            "Choose OpenAI, DeepSeek, Claude, or an OpenAI-compatible API "
            "before testing."
        )
    key_value = str(api_key or "").strip()
    if not key_value:
        raise AIProviderError("An API key is required for the connection test.")
    model_value = str(model or "").strip()
    if not model_value:
        raise AIProviderError("A provider model name is required.")
    timeout_value = float(timeout_seconds)
    if timeout_value <= 0 or timeout_value > 600:
        raise AIProviderError("Timeout must be between 1 and 600 seconds.")

    base_value = _validate_provider_base_url(base_url)
    if provider_name in (AI_PROVIDER_OPENAI, AI_PROVIDER_ANTHROPIC):
        endpoint_path = "/v1/models"
    else:
        endpoint_path = "/models"
    endpoint = _provider_endpoint(base_value, endpoint_path)
    if provider_name == AI_PROVIDER_ANTHROPIC:
        headers = {
            "x-api-key": key_value,
            "anthropic-version": "2023-06-01",
            "Accept": "application/json",
        }
    else:
        headers = {
            "Authorization": f"Bearer {key_value}",
            "Accept": "application/json",
        }

    if transport is not None:
        response = transport(
            endpoint,
            None,
            headers,
            timeout_value,
        )
    else:
        request = urllib_request.Request(
            endpoint,
            headers=headers,
            method="GET",
        )
        response = _read_json_response(request, timeout_value)

    if not isinstance(response, dict):
        raise AIProviderError(
            "The provider returned an invalid connection-test response."
        )

    model_ids = []
    for item in response.get("data", []):
        if isinstance(item, dict) and item.get("id"):
            model_ids.append(str(item["id"]))

    return {
        "provider": provider_name,
        "model": model_value,
        "base_url": base_value,
        "authenticated": True,
        "model_list_available": bool(model_ids),
        "model_available": (
            model_value in model_ids if model_ids else None
        ),
        "available_model_count": len(model_ids),
    }


def _post_json_request(url, payload, headers, timeout_seconds):
    encoded = json.dumps(payload).encode("utf-8")
    request = urllib_request.Request(
        url,
        data=encoded,
        headers=headers,
        method="POST",
    )
    try:
        with urllib_request.urlopen(
            request,
            timeout=timeout_seconds,
        ) as response:
            response_text = response.read().decode("utf-8")
    except urllib_error.HTTPError as exc:
        error_body = exc.read().decode("utf-8", errors="replace")
        try:
            parsed_error = json.loads(error_body)
            detail = (
                parsed_error.get("error", {}).get("message")
                or parsed_error.get("message")
                or error_body
            )
        except Exception:
            detail = error_body
        raise AIProviderError(
            f"Provider returned HTTP {exc.code}: {detail[:800]}"
        ) from exc
    except urllib_error.URLError as exc:
        raise AIProviderError(
            f"Could not connect to the AI provider: {exc.reason}"
        ) from exc

    try:
        return json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise AIProviderError(
            "The provider returned a non-JSON HTTP response."
        ) from exc


def _extract_openai_response_text(response):
    if response.get("status") == "incomplete":
        reason = response.get("incomplete_details", {}).get(
            "reason",
            "unknown reason",
        )
        raise AIProviderError(f"OpenAI response was incomplete: {reason}")

    text_parts = []
    refusals = []
    for output_item in response.get("output", []):
        if output_item.get("type") != "message":
            continue
        for content in output_item.get("content", []):
            if content.get("type") == "output_text":
                text_parts.append(content.get("text", ""))
            elif content.get("type") == "refusal":
                refusals.append(content.get("refusal", "Request refused."))

    if refusals:
        raise AIProviderError("OpenAI refused the request: " + " ".join(refusals))
    text_value = "\n".join(part for part in text_parts if part).strip()
    if not text_value:
        raise AIProviderError("OpenAI returned no structured recommendation.")
    return text_value


def _extract_deepseek_response_text(response):
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise AIProviderError("DeepSeek returned no completion choices.")
    reason = choices[0].get("finish_reason")
    if reason in ("length", "max_tokens"):
        raise AIProviderError("AI response was truncated at the output-token limit (finish_reason=length).")
    if reason == "content_filter":
        raise AIProviderError("AI response was blocked by the provider content filter.")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise AIProviderError("DeepSeek returned empty JSON content.")
    return content.strip()


def _extract_anthropic_response_text(response):
    """Extract concatenated text blocks from an Anthropic Messages response."""
    if not isinstance(response, dict):
        raise AIProviderError("Claude returned an invalid response object.")
    if response.get("type") == "error" or response.get("error"):
        error = response.get("error") or {}
        detail = error.get("message") if isinstance(error, dict) else str(error)
        raise AIProviderError("Claude returned an error: " + str(detail))
    if response.get("stop_reason") in ("max_tokens", "model_context_window_exceeded"):
        raise AIProviderError("Claude response was incomplete: " + response["stop_reason"])
    if response.get("stop_reason") == "refusal":
        raise AIProviderError("Claude refused the request.")
    text_parts = []
    for item in response.get("content", []):
        if isinstance(item, dict) and item.get("type") == "text":
            value = item.get("text")
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
    text_value = "\n".join(text_parts).strip()
    if not text_value:
        raise AIProviderError("Claude returned no usable text content.")
    return text_value


def _anthropic_request_headers(api_key):
    return {
        "x-api-key": str(api_key).strip(),
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
