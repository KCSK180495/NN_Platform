"""Plotting for NN Training Studio."""

from pathlib import Path
import ast
import json
import pandas as pd
import shutil
import subprocess
import sys
import tempfile
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
    AI_PLOT_SCHEMA,
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_OPENAI,
    OFFICIAL_PROVIDER_SETTINGS,
    PLOT_CODE_BLOCKED_NAMES,
)
from nn_training_studio.results import (
    _result_json_default,
)


def _normalise_result_context(context):
    """Return a predictable, copy-safe context for the Results Studio."""
    if not isinstance(context, dict):
        raise ValueError("The result context provider returned invalid data.")
    tables = {}
    for name, value in (context.get("tables") or {}).items():
        safe_name = str(name).strip()
        if not safe_name:
            continue
        if isinstance(value, pd.DataFrame):
            tables[safe_name] = value.copy()
        elif value is not None:
            tables[safe_name] = pd.DataFrame(value)
    if not tables:
        raise ValueError(
            "No result table is available. Run prediction or evaluation first."
        )
    return {
        "title": str(context.get("title") or "Custom Model Result"),
        "task_type": str(context.get("task_type") or "Unknown"),
        "model_type": str(context.get("model_type") or "Unknown"),
        "tables": tables,
        "metrics": dict(context.get("metrics") or {}),
        "metadata": dict(context.get("metadata") or {}),
        "class_names": list(context.get("class_names") or []),
        "source": str(context.get("source") or "Loaded model results"),
    }


def build_result_context_schema(context):
    """Describe result data without exposing raw rows to an AI provider."""
    normalised = _normalise_result_context(context)
    table_schema = {}
    for name, frame in normalised["tables"].items():
        table_schema[name] = {
            "rows": int(len(frame)),
            "columns": [
                {
                    "name": str(column),
                    "dtype": str(frame[column].dtype),
                    "missing": int(frame[column].isna().sum()),
                    "unique": int(frame[column].nunique(dropna=True)),
                }
                for column in frame.columns
            ],
        }
    return {
        "title": normalised["title"],
        "task_type": normalised["task_type"],
        "model_type": normalised["model_type"],
        "source": normalised["source"],
        "tables": table_schema,
        "metrics": normalised["metrics"],
        "class_names": normalised["class_names"],
        "privacy": (
            "Schema, row counts, data types, missing counts, unique counts, "
            "metrics, and class names only. No raw result rows are included."
        ),
    }


def validate_custom_plot_code(code, available_tables=None):
    """
    Validate the manual/AI plotting contract before isolated execution.

    This is a strong accidental-misuse guard, not a security boundary for
    hostile code. Execution also occurs in a child process with a timeout and
    receives only copied result tables.
    """
    code = str(code or "")
    if not code.strip():
        raise ValueError("The plotting code editor is empty.")
    if len(code) > 30000:
        raise ValueError("Plotting code is limited to 30,000 characters.")
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise ValueError(
            f"Python syntax error on line {exc.lineno}: {exc.msg}"
        ) from exc

    blocked_nodes = (
        ast.Import,
        ast.ImportFrom,
        ast.Global,
        ast.Nonlocal,
        ast.ClassDef,
        ast.AsyncFunctionDef,
        ast.Await,
        ast.Yield,
        ast.YieldFrom,
        ast.Delete,
    )
    for node in ast.walk(tree):
        if isinstance(node, blocked_nodes):
            raise ValueError(
                f"{type(node).__name__} is not allowed in custom plots."
            )
        if isinstance(node, ast.Name):
            if (
                node.id in PLOT_CODE_BLOCKED_NAMES
                or node.id.startswith("__")
            ):
                raise ValueError(
                    f"The name '{node.id}' is not allowed in custom plots."
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError(
                f"Private/dunder attribute access '.{node.attr}' is not allowed."
            )
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in PLOT_CODE_BLOCKED_NAMES:
                raise ValueError(
                    f"The call '{node.func.id}(...)' is not allowed."
                )

    for top_level_node in tree.body:
        if not isinstance(top_level_node, ast.FunctionDef):
            raise ValueError(
                "Only function definitions are allowed at the top level. "
                "Put all plotting work inside create_plot(context)."
            )
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "create_plot"
    ]
    if len(functions) != 1:
        raise ValueError(
            "Define exactly one function named create_plot(context)."
        )
    function = functions[0]
    if (
        len(function.args.args) != 1
        or function.args.vararg is not None
        or function.args.kwarg is not None
    ):
        raise ValueError("create_plot must accept exactly one argument: context.")

    table_names = set(available_tables or [])
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.value, ast.Subscript)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "context"
        ):
            # Detailed column validation remains a runtime check because code
            # can choose table/column names dynamically.
            pass
    return {
        "valid": True,
        "function": "create_plot",
        "available_tables": sorted(table_names),
        "line_count": len(code.splitlines()),
    }


def _plot_runner_source():
    return r'''
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import seaborn as sns
except Exception:
    sns = None

work = Path(sys.argv[1])
manifest = json.loads((work / "context.json").read_text(encoding="utf-8"))
tables = {
    name: pd.read_csv(work / relative_path)
    for name, relative_path in manifest["table_files"].items()
}
context = {
    "tables": tables,
    "results": tables.get("results", next(iter(tables.values()))),
    "input_data": tables.get("input_data"),
    "training_history": tables.get("training_history"),
    "detection_results": tables.get("detection_results"),
    "metrics": manifest.get("metrics", {}),
    "metadata": manifest.get("metadata", {}),
    "task_type": manifest.get("task_type"),
    "model_type": manifest.get("model_type"),
    "class_names": manifest.get("class_names", []),
    "pd": pd,
    "np": np,
    "plt": plt,
    "sns": sns,
}
safe_builtins = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "dict": dict,
    "enumerate": enumerate,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "range": range,
    "round": round,
    "set": set,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "zip": zip,
}
namespace = {"__builtins__": safe_builtins}
code = (work / "plot_code.py").read_text(encoding="utf-8")
exec(compile(code, "plot_code.py", "exec"), namespace)
figure = namespace["create_plot"](context)
if figure is None:
    figure = plt.gcf()
if not hasattr(figure, "savefig"):
    raise TypeError("create_plot(context) must return a Matplotlib Figure.")
figure.savefig(work / "plot.png", dpi=180, bbox_inches="tight")
figure.savefig(work / "plot.svg", bbox_inches="tight")
plt.close(figure)
(work / "execution_summary.json").write_text(
    json.dumps(
        {
            "status": "succeeded",
            "plot_png": "plot.png",
            "plot_svg": "plot.svg",
        },
        indent=2,
    ),
    encoding="utf-8",
)
'''


def execute_custom_plot_code(code, context, timeout_seconds=30):
    """Run validated plotting code in a separate Python process."""
    normalised = _normalise_result_context(context)
    validate_custom_plot_code(code, normalised["tables"].keys())
    output_directory = Path(
        tempfile.mkdtemp(prefix="nn_custom_result_")
    )
    try:
        table_files = {}
        for index, (name, frame) in enumerate(
            normalised["tables"].items(),
            start=1,
        ):
            file_name = f"table_{index:02d}.csv"
            frame.to_csv(output_directory / file_name, index=False)
            table_files[name] = file_name
        manifest = {
            "task_type": normalised["task_type"],
            "model_type": normalised["model_type"],
            "metrics": normalised["metrics"],
            "metadata": normalised["metadata"],
            "class_names": normalised["class_names"],
            "table_files": table_files,
        }
        (output_directory / "context.json").write_text(
            json.dumps(manifest, indent=2, default=_result_json_default),
            encoding="utf-8",
        )
        (output_directory / "plot_code.py").write_text(
            code,
            encoding="utf-8",
        )
        (output_directory / "_plot_runner.py").write_text(
            _plot_runner_source(),
            encoding="utf-8",
        )
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(output_directory / "_plot_runner.py"),
                    str(output_directory),
                ],
                capture_output=True,
                text=True,
                timeout=float(timeout_seconds),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Plot execution exceeded the {timeout_seconds}-second limit."
            ) from exc
        if completed.returncode != 0:
            details = (completed.stderr or completed.stdout).strip()
            raise RuntimeError(
                "Custom plot execution failed:\n" + details[-3000:]
            )
        png_path = output_directory / "plot.png"
        svg_path = output_directory / "plot.svg"
        if not png_path.is_file() or not svg_path.is_file():
            raise RuntimeError(
                "The plotting process finished without creating PNG and SVG."
            )
        (output_directory / "_plot_runner.py").unlink(missing_ok=True)
        return str(output_directory)
    except Exception:
        shutil.rmtree(output_directory, ignore_errors=True)
        raise


def request_ai_plot_recipe(
    settings_owner,
    context_schema,
    user_request,
):
    """Request reviewed plotting code from the configured provider."""
    if settings_owner is None:
        raise AIProviderError("Application AI settings are unavailable.")
    provider_name = settings_owner.ai_provider_var.get()
    if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
        raise AIProviderError(
            "Connect OpenAI, DeepSeek, Claude, or an OpenAI-compatible API "
            "in Settings before generating a plot."
        )
    api_key = settings_owner.resolve_ai_api_key(provider_name)
    defaults = OFFICIAL_PROVIDER_SETTINGS[provider_name]
    if not api_key:
        raise AIProviderError(
            "No API key is available. Configure "
            f"{defaults['api_key_environment']} or connect in Settings."
        )
    model_name = settings_owner.ai_model_var.get().strip()
    base_url = _validate_provider_base_url(
        settings_owner.ai_base_url_var.get()
    )
    timeout_seconds = float(settings_owner.ai_timeout_var.get())
    prompt = {
        "request": str(user_request or "").strip(),
        "result_context_schema": context_schema,
        "contract": {
            "entry_point": "def create_plot(context):",
            "must_return": "a Matplotlib Figure",
            "available_context_keys": [
                "tables",
                "results",
                "input_data",
                "training_history",
                "detection_results",
                "metrics",
                "metadata",
                "task_type",
                "model_type",
                "class_names",
                "pd",
                "np",
                "plt",
                "sns",
            ],
            "restrictions": [
                "No imports.",
                "No files, network, operating-system, or subprocess access.",
                "Use only columns listed in the supplied schema.",
                "Do not execute code at module level.",
                "Keep the graph readable and label every axis.",
            ],
        },
    }
    system_prompt = (
        "You generate one locally reviewable Matplotlib result visualization "
        "for a no-code neural-network application. Return JSON only and follow "
        "the schema exactly. The code must define create_plot(context), must "
        "return a Matplotlib Figure, must not import anything, and must use "
        "only the supplied result schema. Never invent unavailable columns."
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
                            prompt,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "text": {
                    "format": {
                        "type": "json_schema",
                        "name": "nn_custom_result_plot",
                        "strict": True,
                        "schema": AI_PLOT_SCHEMA,
                    }
                },
                "max_output_tokens": 5000,
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
                    + json.dumps(AI_PLOT_SCHEMA, separators=(",", ":"))
                ),
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps(
                            prompt,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    }
                ],
                "max_tokens": 5000,
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
                                AI_PLOT_SCHEMA,
                                separators=(",", ":"),
                            )
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            prompt,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        ),
                    },
                ],
                "response_format": {"type": "json_object"},
                "max_tokens": 5000,
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
        recipe = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AIProviderError(
            "The provider returned invalid plot-recipe JSON."
        ) from exc
    for key in AI_PLOT_SCHEMA["required"]:
        if key not in recipe:
            raise AIProviderError(
                f"The AI plot recipe is missing '{key}'."
            )
    if recipe["table"] not in context_schema["tables"]:
        raise AIProviderError(
            f"The AI selected unavailable table '{recipe['table']}'."
        )
    available_columns = {
        item["name"]
        for item in context_schema["tables"][recipe["table"]]["columns"]
    }
    missing = [
        column
        for column in recipe["required_columns"]
        if column not in available_columns
    ]
    if missing:
        raise AIProviderError(
            "The AI selected unavailable columns: " + ", ".join(missing)
        )
    validate_custom_plot_code(
        recipe["code"],
        context_schema["tables"].keys(),
    )
    recipe["provider"] = provider_name
    recipe["model"] = model_name
    recipe["data_sent"] = context_schema["privacy"]
    return recipe
