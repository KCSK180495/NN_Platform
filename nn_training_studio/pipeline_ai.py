"""Ask a configured provider for a locally validated declarative recipe."""

import json

from .ai_transport import (
    _anthropic_request_headers,
    _extract_anthropic_response_text,
    _extract_deepseek_response_text,
    _extract_openai_response_text,
    _provider_endpoint,
)
from .constants import AI_PROVIDER_ANTHROPIC, AI_PROVIDER_OPENAI
from .processing_pipeline import OPERATIONS, SCHEMA, validate_recipe


def draft_recipe(provider, provider_name, goal, inputs, rate):
    operations = {op: params for op, params in OPERATIONS.items() if op not in ("python", "sscn")}
    system = (
        "Draft a signal processing recipe as one JSON object. Use exactly the recipe structure "
        "in the supplied example. Keep the supplied input names, units and sample rate. "
        "Each step must read existing columns and produce unique new columns. The arithmetic "
        "steps add, subtract, multiply and divide take two ordered inputs. Other steps take one input. "
        "Numeric fitted parameters are learned locally. Never generate code or invent measurements. "
        "Use only the supplied operations and parameter names."
    )
    brief = json.dumps({
        "goal": goal,
        "example": {
            "schema": SCHEMA, "name": "Processing pipeline", "sample_rate_hz": rate,
            "inputs": inputs, "steps": [
                {"op": "ema", "inputs": [next(iter(inputs))], "outputs": ["smoothed"],
                 "params": {"span": 5}},
            ],
            "outputs": ["smoothed"],
        },
        "operation_defaults": operations,
    })
    messages = [{"role": "system", "content": system}, {"role": "user", "content": brief}]
    headers = {"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"}
    if provider_name == AI_PROVIDER_OPENAI:
        endpoint = "/v1/responses"
        payload = {"model": provider.model, "input": messages, "max_output_tokens": 4000,
                   "store": False, "text": {"format": {"type": "json_object"}}}
        extract = _extract_openai_response_text
    elif provider_name == AI_PROVIDER_ANTHROPIC:
        endpoint = "/v1/messages"
        payload = {"model": provider.model, "messages": messages[1:], "system": system, "max_tokens": 4000}
        headers = _anthropic_request_headers(provider.api_key)
        extract = _extract_anthropic_response_text
    else:
        endpoint = "/chat/completions"
        payload = {"model": provider.model, "messages": messages, "max_tokens": 4000,
                   "response_format": {"type": "json_object"}, "stream": False}
        extract = _extract_deepseek_response_text
    response = provider.transport(_provider_endpoint(provider.base_url, endpoint), payload,
                                  headers, provider.timeout_seconds)
    raw = extract(response).strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = raw.partition("\n")[2][:-3].strip()
    recipe = validate_recipe(json.loads(raw))
    if recipe["inputs"] != inputs or recipe["sample_rate_hz"] != rate:
        raise ValueError("The draft changed the declared inputs or sample rate.")
    if any(step["op"] not in operations for step in recipe["steps"]):
        raise ValueError("The draft contains an operation that requires manual configuration.")
    return recipe
