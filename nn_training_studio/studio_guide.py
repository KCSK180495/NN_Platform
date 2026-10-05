"""Studio guide for NN Training Studio."""

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
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_OPENAI,
    AI_STUDIO_GUIDE_SCHEMA,
    GUIDE_ACTION_DEPLOY,
    GUIDE_ACTION_MODEL_APPLICATION,
    GUIDE_ACTION_NONE,
    GUIDE_ACTION_PROJECTS,
    GUIDE_ACTION_SETTINGS,
    GUIDE_ALLOWED_ACTIONS,
    OFFICIAL_PROVIDER_SETTINGS,
    STUDIO_CAPABILITY_CATALOGUE,
    WORKSPACE_ANNOTATE,
    WORKSPACE_DETECTION,
    WORKSPACE_FILTER,
    WORKSPACE_IMAGE,
    WORKSPACE_SIGNAL,
)


def _offline_studio_guide_answer(question, current_context="home"):
    """Return deterministic navigation help without using an AI provider."""
    text = " ".join(str(question or "").strip().lower().split())
    context = str(current_context or "home")

    def result(answer, checklist, action):
        return {
            "answer": answer,
            "checklist": checklist,
            "recommended_action": action,
            "mode": "Offline Guide",
        }

    if not text:
        return result(
            "Tell me what data you have and what result you want. For example: "
            "'I have CSV motor currents and want to predict the next signal.'",
            [
                "Mention the data format: CSV/Excel, image folders, or YOLO labels.",
                "Mention the output: class, value, future signal, reconstruction, or boxes.",
            ],
            GUIDE_ACTION_NONE,
        )

    if any(
        phrase in text
        for phrase in (
            "show workflow",
            "main workflow",
            "how does nn studio work",
            "where do i start",
            "help me choose",
        )
    ):
        return result(
            "Start from the output you need. Create & Train builds a new model; "
            "Apply & Validate uses an existing model; Prepare Data filters "
            "signals or annotates 1D/2D data before training; Deploy & Integrate connects a finished "
            "model to software or equipment; Manage Studio handles checkpoints "
            "and settings. Tell me your data format and desired output, and I "
            "can select one workspace.",
            [
                "Identify the data format.",
                "Choose the desired output: class, value, future signal, reconstruction, or boxes.",
                "Train a new model or load an existing model.",
                "Evaluate, preview, customise, and export the results.",
            ],
            GUIDE_ACTION_NONE,
        )

    if any(
        phrase in text
        for phrase in (
            "deploy",
            "deployment",
            "integrate model",
            "implement api",
            "integrate api",
            "add chatgpt",
            "implement chatgpt",
            "integrate chatgpt",
            "add deepseek",
            "implement deepseek",
            "integrate deepseek",
            "add claude",
            "implement claude",
            "integrate claude",
            "implement openai",
            "integrate openai",
            "ai assistant in my app",
            "implementation code",
            "my code",
            "source code",
            "analyse my code",
            "analyze my code",
            "copy and paste",
            "vendor sdk",
            "hardware sdk",
            "use in my application",
            "connect model to",
            "rest api",
            "docker",
            "raspberry pi",
            "serial device",
            "mqtt",
            "modbus",
            "opc ua",
            "plc",
            "iot",
        )
    ):
        return result(
            "Use the five-step Deploy & Integrate flow. Start by selecting your "
            "existing project and describing the result in plain language. Then "
            "choose whether the application should use a trained model, call an "
            "AI provider API such as OpenAI, DeepSeek, or Claude, or use both. "
            "NN Studio plans the mapping, validates the selected path, generates "
            "complete code, and lets the AI coding assistant propose a reviewed "
            "project patch or copy-and-paste files.",
            [
                "Upload a project folder or select only the code files to analyse.",
                "Describe the application and desired AI feature in one sentence.",
                "Choose Trained Model, AI Provider API, or Both.",
                "Test compatibility and, for a loaded model, validate sample input.",
                "Review generated code and the AI-proposed changes before export.",
            ],
            GUIDE_ACTION_DEPLOY,
        )

    if any(
        phrase in text
        for phrase in (
            "load model",
            "existing model",
            "use model",
            "apply model",
            "inference",
            "predict new",
            "preview result",
            "custom result",
            "plot result",
            "validation result",
        )
    ):
        return result(
            "Use Model Application. It identifies the package type, checks new "
            "input data, runs prediction or evaluation, previews results, and "
            "opens the Custom Results Studio for built-in, manual, or AI plots.",
            [
                "Select the saved model package or detector weights.",
                "Review its input and preprocessing requirements.",
                "Load compatible new data and run prediction or evaluation.",
                "Preview, customise, and export the results.",
            ],
            GUIDE_ACTION_MODEL_APPLICATION,
        )

    if any(
        phrase in text
        for phrase in (
            "annotate",
            "annotation",
            "label my data",
            "label signal",
            "label image",
            "event point",
            "fault interval",
            "draw box",
            "pre-annotation",
            "class balance",
        )
    ):
        return result(
            "Use Annotate & Prepare Data before training. Load a CSV/Excel "
            "signal or image folder, create a label scheme, add intervals, "
            "events, image classes, or bounding boxes, and review any local or "
            "AI suggestions. Only approved labels are transferred to training.",
            [
                "Load 1D signal/table data or 2D images.",
                "Create or request a suitable label scheme.",
                "Add manual labels or generate pending suggestions.",
                "Approve, edit, or reject every suggested annotation.",
                "Run the quality check, then export or continue to training.",
            ],
            WORKSPACE_ANNOTATE,
        )

    if any(
        phrase in text
        for phrase in (
            "object detection",
            "bounding box",
            "bounding boxes",
            "yolo",
            "detect object",
            "video detection",
            "webcam",
        )
    ):
        return result(
            "Choose Object Detection when each image can contain one or more "
            "objects and you need class names plus bounding boxes. Prepare a "
            "YOLO data.yaml file with matching images and label text files.",
            [
                "Arrange train/validation images and YOLO label files.",
                "Confirm class names in data.yaml.",
                "Validate boxes before training.",
                "Choose Auto, CPU, or a detected GPU and start from pretrained weights.",
            ],
            WORKSPACE_DETECTION,
        )

    if any(
        phrase in text
        for phrase in (
            "image classification",
            "photo classification",
            "classify image",
            "classify photo",
            "image folder",
            "photos",
            "pictures",
        )
    ):
        return result(
            "Choose Image Classification when every complete image has one "
            "class. Put images in class-named folders such as Normal, SEF, "
            "DEF, and MEF.",
            [
                "Create one folder for each class.",
                "Check that every class has enough images.",
                "Preview the split and training-only augmentation.",
                "Select CNN2D or transfer learning and train.",
            ],
            WORKSPACE_IMAGE,
        )

    if any(
        phrase in text
        for phrase in (
            "filter",
            "denoise",
            "smooth",
            "moving average",
            "kalman",
            "butterworth",
            "clean data",
        )
    ):
        return result(
            "Use Filter & Export when you want to clean or inspect signals "
            "without training. You can select only the affected columns, "
            "preview before/after plots, save filtered CSV/Excel data, and "
            "reuse the filter preset later.",
            [
                "Load CSV or Excel data.",
                "Select numeric signal columns only.",
                "Preview the filter and verify that important transients remain.",
                "Export the data or continue directly to signal training.",
            ],
            WORKSPACE_FILTER,
        )

    if any(
        phrase in text
        for phrase in (
            "forecast",
            "future signal",
            "next signal",
            "next sample",
            "predict signal",
            "time series",
        )
    ):
        return result(
            "Use Signal / Tabular Training and select Time-series Forecasting. "
            "The model learns from chronological windows and predicts the next "
            "one or more samples for the selected target signals.",
            [
                "Load time-ordered CSV or Excel data.",
                "Select signal inputs and forecasting targets.",
                "Choose window size, stride, and forecast horizon.",
                "Use a chronological split, then review the forecasting timeline.",
            ],
            WORKSPACE_SIGNAL,
        )

    if any(
        phrase in text
        for phrase in (
            "anomaly",
            "reconstruct",
            "reconstruction",
            "autoencoder",
        )
    ):
        return result(
            "Use Signal / Tabular Training and select Autoencoder / Anomaly "
            "Detection. It reconstructs the input signal and uses "
            "reconstruction error as the anomaly score.",
            [
                "Load representative normal or mixed-condition signal data.",
                "Select the input signals to reconstruct.",
                "Choose row-based or sequence-based preprocessing.",
                "Review reconstruction error and set an anomaly threshold.",
            ],
            WORKSPACE_SIGNAL,
        )

    if any(
        phrase in text
        for phrase in (
            "regression",
            "predict value",
            "predict speed",
            "predict torque",
            "predict temperature",
            "continuous value",
            "multiple values",
            "u v w",
            "ia ib ic",
        )
    ) or (
        "predict" in text
        and any(
            target in text
            for target in (
                "speed",
                "torque",
                "temperature",
                "voltage",
                "current",
                "power",
            )
        )
    ):
        return result(
            "Use Signal / Tabular Training. Select Regression for one "
            "continuous target or Multi-output Regression for several values "
            "such as U/V/W, Ia/Ib/Ic, speed, or torque.",
            [
                "Load CSV or Excel data.",
                "Choose input columns and one or more numeric target columns.",
                "Confirm scaling and original-unit inverse scaling.",
                "Evaluate MAE, RMSE, R², and actual-versus-predicted plots.",
            ],
            WORKSPACE_SIGNAL,
        )

    if any(
        phrase in text
        for phrase in (
            "classification",
            "classify",
            "fault class",
            "normal sef",
            "normal or fault",
            "condition",
            "label",
        )
    ):
        return result(
            "Use Signal / Tabular Training for labelled sensor tables and "
            "select Classification. Choose the signal columns as inputs and "
            "the condition column as the target label.",
            [
                "Load a CSV or Excel dataset containing labels.",
                "Check class balance and missing values.",
                "Select input signals and one label column.",
                "Review accuracy, weighted F1, and the confusion matrix.",
            ],
            WORKSPACE_SIGNAL,
        )

    if any(
        phrase in text
        for phrase in (
            "checkpoint",
            "project",
            "resume training",
            "best model",
            "training run",
        )
    ):
        return result(
            "Open Projects & Checkpoints to inspect saved runs, best-model "
            "checkpoints, and recovery states. Use Model Application when you "
            "want to evaluate or predict with a finished package.",
            [
                "Locate the required project or run.",
                "Review its status and best validation checkpoint.",
                "Resume recovery or open the completed model package.",
            ],
            GUIDE_ACTION_PROJECTS,
        )

    if any(
        phrase in text
        for phrase in (
            "api",
            "openai",
            "deepseek",
            "gpu",
            "cpu",
            "device",
            "setting",
        )
    ):
        return result(
            "Open Settings to configure the AI provider and guide preferences. "
            "Training-device choices remain inside each training or inference "
            "workspace so the selected hardware is recorded with that run.",
            [
                "Configure and test an AI provider only if AI help is required.",
                "Keep Offline Guide available for navigation without an API key.",
                "Choose Auto, CPU, or a detected GPU inside the relevant workspace.",
            ],
            GUIDE_ACTION_SETTINGS,
        )

    return result(
        "I can guide you to signal/value training, image classification, "
        "object detection, filtering, model application, deployment, projects, "
        "or settings. Your current page is "
        f"'{context}'. Tell me your data format and desired output for a more "
        "specific recommendation.",
        [
            "What data do you have?",
            "What should the model output?",
            "Are labels or target values available?",
        ],
        GUIDE_ACTION_NONE,
    )


def request_ai_studio_guide(settings_owner, question, current_context):
    """Ask the configured provider for navigation help without user data."""
    if settings_owner is None:
        raise AIProviderError("Application AI settings are unavailable.")
    provider_name = settings_owner.ai_provider_var.get()
    if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
        raise AIProviderError(
            "Connect OpenAI, DeepSeek, Claude, or an OpenAI-compatible API "
            "in Settings, or turn off AI mode to "
            "use the offline guide."
        )
    api_key = settings_owner.resolve_ai_api_key(provider_name)
    if not api_key:
        raise AIProviderError(
            "No provider API key is available. Configure it in Settings, or "
            "use the offline guide."
        )
    model_name = settings_owner.ai_model_var.get().strip()
    base_url = _validate_provider_base_url(
        settings_owner.ai_base_url_var.get()
    )
    timeout_seconds = float(settings_owner.ai_timeout_var.get())
    user_payload = {
        "question": str(question or "").strip(),
        "current_page": str(current_context or "home"),
        "studio_capabilities": STUDIO_CAPABILITY_CATALOGUE,
        "privacy": (
            "No dataset rows, model files, predictions, results, credentials, "
            "or conversation history are included."
        ),
    }
    system_prompt = (
        "You are the navigation guide inside NN Studio, a no-code desktop "
        "machine-learning application. Help the user select an existing "
        "workspace and explain the immediate workflow in plain language. If "
        "the user asks to analyse code or implement a model or provider API, "
        "turn the request into a short actionable plan and recommend Deploy & "
        "Integrate; do not require the user to know technical terminology. Do "
        "not claim unsupported capabilities. Return JSON only. Recommend "
        "'none' when the question does not justify opening a workspace. Never "
        "ask for API keys, raw data, model files, personal data, or secrets."
    )
    if provider_name == AI_PROVIDER_OPENAI:
        response = _post_json_request(
            _provider_endpoint(base_url, "/v1/responses"),
            {
                "model": model_name,
                "input": [
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": json.dumps(
                            user_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "nn_studio_guide",
                        "strict": True,
                        "schema": AI_STUDIO_GUIDE_SCHEMA,
                    }
                },
                "max_output_tokens": 2200,
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
                "system": (
                    system_prompt
                    + "\nReturn one JSON object only. Exact JSON schema:\n"
                    + json.dumps(
                        AI_STUDIO_GUIDE_SCHEMA,
                        separators=(",", ":"),
                    )
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            user_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                ],
                "max_tokens": 2200,
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
                        "content": (
                            system_prompt
                            + "\nExact JSON schema:\n"
                            + json.dumps(
                                AI_STUDIO_GUIDE_SCHEMA,
                                separators=(",", ":"),
                            )
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            user_payload,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 2200,
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
        answer = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AIProviderError(
            "The provider returned invalid guide-response JSON."
        ) from exc
    if set(answer) != set(AI_STUDIO_GUIDE_SCHEMA["required"]):
        raise AIProviderError(
            "The provider returned an incomplete guide response."
        )
    if (
        not isinstance(answer["answer"], str)
        or not answer["answer"].strip()
        or not isinstance(answer["checklist"], list)
        or not all(
            isinstance(item, str) and item.strip()
            for item in answer["checklist"]
        )
        or answer["recommended_action"] not in GUIDE_ALLOWED_ACTIONS
    ):
        raise AIProviderError(
            "The provider returned unsupported guide-response values."
        )
    answer["answer"] = answer["answer"].strip()[:1800]
    answer["checklist"] = [
        item.strip()[:240] for item in answer["checklist"][:6]
    ]
    answer["mode"] = f"AI Guide — {provider_name}"
    return answer
