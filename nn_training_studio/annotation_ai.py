"""Annotation ai for NN Training Studio."""

import json
from nn_training_studio.ai_transport import (
    _anthropic_request_headers,
    _extract_anthropic_response_text,
    _extract_deepseek_response_text,
    _extract_openai_response_text,
    _post_json_request,
    _provider_endpoint,
    _validate_provider_base_url,
)
from nn_training_studio.ai_validation import (
    AIProviderError,
)
from nn_training_studio.constants import (
    AI_ANNOTATION_PLAN_SCHEMA,
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_OPENAI,
    OFFICIAL_PROVIDER_SETTINGS,
)


def validate_ai_annotation_plan(plan):
    """Validate a provider annotation plan before displaying or applying it."""
    if not isinstance(plan, dict):
        raise AIProviderError("The annotation plan must be a JSON object.")
    required = AI_ANNOTATION_PLAN_SCHEMA["required"]
    missing = [key for key in required if key not in plan]
    if missing:
        raise AIProviderError(
            "The annotation plan is missing: " + ", ".join(missing)
        )
    allowed_types = set(
        AI_ANNOTATION_PLAN_SCHEMA["properties"]["annotation_type"]["enum"]
    )
    if plan["annotation_type"] not in allowed_types:
        raise AIProviderError("The provider selected an unsupported annotation type.")
    cleaned = {
        "summary": str(plan["summary"]).strip()[:1200],
        "annotation_type": plan["annotation_type"],
    }
    for key, maximum in (
        ("suggested_labels", 20),
        ("local_methods", 8),
        ("review_checks", 8),
        ("warnings", 8),
    ):
        values = plan.get(key)
        if not isinstance(values, list):
            raise AIProviderError(f"'{key}' must be a JSON array.")
        cleaned[key] = []
        for value in values[:maximum]:
            text_value = " ".join(str(value).split()).strip()
            if text_value and text_value not in cleaned[key]:
                cleaned[key].append(text_value[:240])
    if not cleaned["summary"] or not cleaned["suggested_labels"]:
        raise AIProviderError(
            "The annotation plan needs a summary and at least one label."
        )
    if not cleaned["review_checks"]:
        raise AIProviderError(
            "The annotation plan must include at least one human review check."
        )
    return cleaned


def request_ai_annotation_plan(settings_owner, compact_profile, user_goal):
    """Request a plan using compact metadata; raw samples/pixels stay local."""
    if isinstance(settings_owner, dict):
        provider_name = settings_owner.get("provider")
        api_key = str(settings_owner.get("api_key") or "").strip()
        model_name = str(settings_owner.get("model") or "").strip()
        base_url_value = str(settings_owner.get("base_url") or "").strip()
        timeout_value = settings_owner.get("timeout", 90)
    else:
        provider_name = settings_owner.ai_provider_var.get()
        api_key = settings_owner.resolve_ai_api_key(provider_name)
        model_name = settings_owner.ai_model_var.get().strip()
        base_url_value = settings_owner.ai_base_url_var.get().strip()
        timeout_value = settings_owner.ai_timeout_var.get() or 90
    if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
        raise AIProviderError(
            "Configure OpenAI, DeepSeek, Claude, or a compatible provider first."
        )
    if not api_key:
        raise AIProviderError("No API key is available for the selected provider.")
    if not model_name:
        raise AIProviderError("Configure a provider model name in Settings.")
    base_url = _validate_provider_base_url(base_url_value)
    timeout_seconds = float(timeout_value)
    payload_context = {
        "user_goal": str(user_goal or "").strip()[:1800],
        "local_profile": compact_profile,
        "privacy": (
            "Only this compact profile is provided. No raw signal samples, "
            "dataset rows, image pixels, credentials, or file contents are sent."
        ),
    }
    system_prompt = (
        "You are an annotation-planning assistant for beginner users. Design a "
        "conservative human-in-the-loop labelling plan. AI output is advisory: "
        "never claim that a condition or fault is confirmed from summary data. "
        "Return JSON matching the supplied schema exactly."
    )
    user_prompt = json.dumps(payload_context, ensure_ascii=False, separators=(",", ":"))
    if provider_name == AI_PROVIDER_OPENAI:
        response = _post_json_request(
            _provider_endpoint(base_url, "/v1/responses"),
            {
                "model": model_name,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "annotation_plan",
                        "strict": True,
                        "schema": AI_ANNOTATION_PLAN_SCHEMA,
                    }
                },
                "max_output_tokens": 3500,
                "store": False,
            },
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout_seconds,
        )
        raw_text = _extract_openai_response_text(response)
    elif provider_name == AI_PROVIDER_ANTHROPIC:
        response = _post_json_request(
            _provider_endpoint(base_url, "/v1/messages"),
            {
                "model": model_name,
                "system": system_prompt + "\nExact schema:\n" + json.dumps(
                    AI_ANNOTATION_PLAN_SCHEMA, separators=(",", ":")
                ),
                "messages": [{"role": "user", "content": user_prompt}],
                "max_tokens": 3500,
                "temperature": 0,
            },
            _anthropic_request_headers(api_key),
            timeout_seconds,
        )
        raw_text = _extract_anthropic_response_text(response)
    else:
        response = _post_json_request(
            _provider_endpoint(base_url, "/chat/completions"),
            {
                "model": model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt + "\nExact schema:\n" + json.dumps(
                            AI_ANNOTATION_PLAN_SCHEMA, separators=(",", ":")
                        ),
                    },
                    {"role": "user", "content": user_prompt},
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 3500,
                "stream": False,
            },
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout_seconds,
        )
        raw_text = _extract_deepseek_response_text(response)
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AIProviderError(
            "The provider returned invalid annotation-plan JSON."
        ) from exc
    plan = validate_ai_annotation_plan(parsed)
    plan["provider"] = provider_name
    plan["model"] = model_name
    plan["data_sent"] = payload_context["privacy"]
    return plan
