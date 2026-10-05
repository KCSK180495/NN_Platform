"""Integration for NN Training Studio."""

from pathlib import Path
import ast
import difflib
import hashlib
import json
import os
import shutil
import zipfile
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
    AI_CODE_BLOCKED_DIRECTORIES,
    AI_CODE_BLOCKED_FILENAMES,
    AI_CODE_INTEGRATION_SCHEMA,
    AI_CODE_INTEGRATION_SYSTEM_PROMPT,
    AI_CODE_MAX_CANDIDATES,
    AI_CODE_MAX_FILE_BYTES,
    AI_CODE_MAX_SELECTED_FILES,
    AI_CODE_MAX_TOTAL_BYTES,
    AI_CODE_SOURCE_EXTENSIONS,
    AI_COMPLETE_PACKAGE_EXCLUDED_DIRECTORIES,
    AI_COMPLETE_PACKAGE_SECRET_FILENAMES,
    AI_COMPLETE_PACKAGE_SECRET_SUFFIXES,
    AI_PROVIDER_ANTHROPIC,
    AI_PROVIDER_OPENAI,
    INTEGRATION_MODE_HYBRID,
    INTEGRATION_MODE_LOCAL_MODEL,
    INTEGRATION_MODE_PROVIDER_API,
    OFFICIAL_PROVIDER_SETTINGS,
    _AI_CODE_SECRET_PATTERN,
)


def _normalise_project_relative_path(path):
    value = str(path or "").replace("\\", "/").strip().lstrip("./")
    candidate = Path(value)
    if not value or candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"Unsafe project-relative path: {path!r}")
    return candidate.as_posix()


def _ai_code_block_reason(relative_path):
    """Return a privacy/safety exclusion reason, or an empty string."""
    try:
        value = _normalise_project_relative_path(relative_path)
    except ValueError as exc:
        return str(exc)
    path = Path(value)
    lower_parts = {part.lower() for part in path.parts[:-1]}
    blocked_directories = {item.lower() for item in AI_CODE_BLOCKED_DIRECTORIES}
    if lower_parts & blocked_directories:
        return "generated dependency, cache, environment, or repository metadata"
    filename = path.name.lower()
    if filename in AI_CODE_BLOCKED_FILENAMES or filename.startswith(".env."):
        return "credential or secret configuration file"
    if any(
        marker in filename
        for marker in ("private_key", "secret", "credential", "keystore")
    ):
        return "filename indicates credentials or secrets"
    if (
        path.suffix.lower() not in AI_CODE_SOURCE_EXTENSIONS
        and filename not in {"dockerfile", "makefile", "procfile"}
    ):
        return "unsupported or non-text source type"
    return ""


def _source_contains_probable_secret(text):
    """Conservatively block files that appear to contain credential values."""
    for match in _AI_CODE_SECRET_PATTERN.finditer(str(text or "")):
        matched_line = match.group(0).lower()
        if any(
            marker in matched_line
            for marker in ("getenv", "process.env", "environ[", "secretmanager")
        ):
            continue
        value = match.group(1).strip()
        if value.lower() not in {
            "your_api_key", "replace_me", "changeme", "example_key",
            "<api_key>", "${api_key}", "os.getenv", "process.env",
        }:
            return True
    return False


def scan_ai_integration_project(project_root):
    """Inspect a project locally and return shareable source candidates."""
    root = Path(project_root).resolve()
    if not root.is_dir():
        raise ValueError("Select an existing application project folder.")
    priority_names = {
        "app.py", "main.py", "server.py", "api.py", "routes.py",
        "package.json", "requirements.txt", "pyproject.toml", "dockerfile",
        "index.js", "index.ts", "app.js", "app.ts", "program.cs",
        "main.cpp", "main.c", "platformio.ini", "readme.md",
    }
    candidates = []
    blocked = []
    for path in root.rglob("*"):
        if len(candidates) >= AI_CODE_MAX_CANDIDATES:
            break
        if not path.is_file() or path.is_symlink():
            continue
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            continue
        reason = _ai_code_block_reason(relative)
        if reason:
            if path.name.lower() in AI_CODE_BLOCKED_FILENAMES:
                blocked.append({"path": relative, "reason": reason})
            continue
        try:
            size = path.stat().st_size
        except OSError:
            continue
        if size > AI_CODE_MAX_FILE_BYTES:
            blocked.append({"path": relative, "reason": "source file is too large"})
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            blocked.append({"path": relative, "reason": "not readable UTF-8 text"})
            continue
        if _source_contains_probable_secret(text):
            blocked.append({"path": relative, "reason": "probable credential value"})
            continue
        depth = len(Path(relative).parts)
        name = path.name.lower()
        recommended = name in priority_names or (
            depth <= 2
            and any(
                token in name
                for token in ("app", "main", "server", "route", "device", "sdk")
            )
        )
        candidates.append(
            {
                "path": relative,
                "size": size,
                "language": path.suffix.lower().lstrip(".") or "text",
                "recommended": bool(recommended),
            }
        )
    candidates.sort(
        key=lambda item: (
            not item["recommended"],
            len(Path(item["path"]).parts),
            item["path"].lower(),
        )
    )
    return {
        "root": str(root),
        "candidates": candidates,
        "blocked": blocked,
        "candidate_limit_reached": len(candidates) >= AI_CODE_MAX_CANDIDATES,
    }


def scan_ai_integration_files(file_paths):
    """Create a privacy-checked project view from user-selected source files."""
    paths = [Path(item).resolve() for item in file_paths if str(item).strip()]
    if not paths:
        raise ValueError("Select at least one source-code file.")
    if len(paths) > AI_CODE_MAX_SELECTED_FILES:
        raise ValueError(
            f"Select no more than {AI_CODE_MAX_SELECTED_FILES} files per request."
        )
    if any(not path.is_file() or path.is_symlink() for path in paths):
        raise ValueError("Every selected item must be an existing source-code file.")
    common = Path(os.path.commonpath([str(path.parent) for path in paths])).resolve()
    candidates = []
    blocked = []
    for path in paths:
        relative = path.relative_to(common).as_posix()
        reason = _ai_code_block_reason(relative)
        try:
            size = path.stat().st_size
        except OSError:
            size = AI_CODE_MAX_FILE_BYTES + 1
        if not reason and size > AI_CODE_MAX_FILE_BYTES:
            reason = "source file is too large"
        text = None
        if not reason:
            try:
                text = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                reason = "not readable UTF-8 text"
        if not reason and _source_contains_probable_secret(text):
            reason = "probable credential value"
        if reason:
            blocked.append({"path": relative, "reason": reason})
            continue
        candidates.append(
            {
                "path": relative,
                "size": size,
                "language": path.suffix.lower().lstrip(".") or "text",
                "recommended": True,
            }
        )
    if not candidates:
        reasons = "; ".join(
            f"{item['path']}: {item['reason']}" for item in blocked
        )
        raise ValueError("No selected file can be shared. " + reasons)
    return {
        "root": str(common),
        "candidates": candidates,
        "blocked": blocked,
        "candidate_limit_reached": False,
        "selection_mode": "individual_files",
    }


def summarise_ai_integration_project(scan):
    """Return a plain-language local project summary without sending source."""
    candidates = list((scan or {}).get("candidates") or [])
    paths = [str(item.get("path", "")) for item in candidates]
    lower_paths = [path.lower() for path in paths]
    suffixes = {Path(path).suffix.lower() for path in paths}
    technologies = []
    if any(path.endswith(("package.json", ".js", ".jsx", ".ts", ".tsx")) for path in lower_paths):
        technologies.append("JavaScript / TypeScript web stack")
    if any(path.endswith((".html", ".css", ".scss", ".vue", ".svelte")) for path in lower_paths):
        technologies.append("browser user interface")
    if ".py" in suffixes or any(path.endswith(("requirements.txt", "pyproject.toml")) for path in lower_paths):
        technologies.append("Python application or service")
    if ".cs" in suffixes or ".csproj" in suffixes:
        technologies.append("C# / .NET application")
    if suffixes & {".c", ".cpp", ".h", ".hpp", ".ino"}:
        technologies.append("C/C++ or embedded hardware code")
    if suffixes & {".java", ".kt", ".kts"}:
        technologies.append("Java / Kotlin application")
    if any(token in path for path in lower_paths for token in ("device", "serial", "sdk", "modbus", "mqtt", "opcua")):
        technologies.append("device or vendor-SDK integration")
    if not technologies:
        technologies.append("general source-code project")
    entry_names = {
        "app.py", "main.py", "server.py", "api.py", "index.js", "index.ts",
        "app.js", "app.ts", "program.cs", "main.cpp", "main.c", "main.ino",
    }
    entry_points = [
        path for path in paths if Path(path).name.lower() in entry_names
    ]
    recommended = [
        str(item["path"]) for item in candidates if item.get("recommended")
    ][:8]
    return {
        "technologies": list(dict.fromkeys(technologies)),
        "entry_points": entry_points[:8],
        "recommended_files": recommended,
        "source_file_count": len(candidates),
        "blocked_file_count": len((scan or {}).get("blocked") or []),
    }


def format_ai_project_summary(summary):
    technologies = ", ".join(summary.get("technologies") or ["Not detected"])
    entries = ", ".join(summary.get("entry_points") or ["Not identified yet"])
    recommended = ", ".join(summary.get("recommended_files") or ["Select manually"])
    return (
        f"Detected locally: {technologies}.\n"
        f"Likely entry points: {entries}.\n"
        f"Recommended files: {recommended}.\n"
        f"Available source files: {summary.get('source_file_count', 0)}; "
        f"privacy-blocked files: {summary.get('blocked_file_count', 0)}."
    )


def build_beginner_integration_request(
    application_type,
    desired_result,
    data_source,
    user_notes="",
    project_summary=None,
    integration_mode=INTEGRATION_MODE_LOCAL_MODEL,
    provider_name="",
    provider_purpose="",
):
    """Build a complete AI brief from plain-language implementation choices."""
    summary = project_summary or {}
    notes = str(user_notes or "").strip()
    detected = ", ".join(summary.get("technologies") or ["not yet detected"])
    entry_points = ", ".join(summary.get("entry_points") or ["identify from code"])
    if integration_mode == INTEGRATION_MODE_PROVIDER_API:
        implementation = (
            f"Implement the configured {provider_name or 'AI provider'} API "
            "into my existing application. No local trained model is required."
        )
    elif integration_mode == INTEGRATION_MODE_HYBRID:
        implementation = (
            f"Implement both the selected trained model and the configured "
            f"{provider_name or 'AI provider'} API into my existing application."
        )
    else:
        implementation = "Implement the selected trained model into my existing application."
    return (
        implementation + "\n"
        f"Integration path: {integration_mode}.\n"
        + (
            f"Provider API purpose: {provider_purpose}.\n"
            if integration_mode != INTEGRATION_MODE_LOCAL_MODEL
            else ""
        )
        + f"Application type: {application_type}.\n"
        f"Desired result: {desired_result}.\n"
        f"Application data source: {data_source}.\n"
        f"Locally detected technology: {detected}.\n"
        f"Likely entry point(s): {entry_points}.\n"
        f"My additional request: {notes or 'No technical details supplied; infer conservatively from the approved code.'}\n\n"
        "Please preserve the application's current behavior and framework. Add the "
        "required model preprocessing or provider request/response handling, input "
        "validation, error handling, and a small test example. Return complete files, "
        "not fragments with omitted sections. Explain in beginner-friendly language "
        "where each file belongs, what values the user must configure, what command "
        "to run, and how to verify one prediction. Never embed an AI-provider key. "
        "For hardware, keep device writes and actuation disabled unless explicitly "
        "requested and protected by an independent safety gate."
    )


def collect_ai_integration_sources(project_root, relative_paths):
    """Read only explicitly selected, locally approved source files."""
    root = Path(project_root).resolve()
    selected = []
    seen = set()
    total_bytes = 0
    if not relative_paths:
        raise ValueError("Select at least one source file to share with the AI.")
    if len(relative_paths) > AI_CODE_MAX_SELECTED_FILES:
        raise ValueError(
            f"Select no more than {AI_CODE_MAX_SELECTED_FILES} files per request."
        )
    for raw_path in relative_paths:
        relative = _normalise_project_relative_path(raw_path)
        if relative in seen:
            continue
        reason = _ai_code_block_reason(relative)
        if reason:
            raise ValueError(f"Cannot share {relative}: {reason}.")
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Source path escapes the project: {relative}") from exc
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Selected source is unavailable: {relative}")
        data = path.read_bytes()
        if len(data) > AI_CODE_MAX_FILE_BYTES:
            raise ValueError(f"Selected source is too large: {relative}")
        total_bytes += len(data)
        if total_bytes > AI_CODE_MAX_TOTAL_BYTES:
            raise ValueError(
                "Selected source exceeds the per-request limit of "
                f"{AI_CODE_MAX_TOTAL_BYTES // 1000} KB."
            )
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"Selected source is not UTF-8 text: {relative}") from exc
        if _source_contains_probable_secret(content):
            raise ValueError(
                f"Cannot share {relative}: a probable credential value was detected."
            )
        selected.append(
            {
                "path": relative,
                "sha256": hashlib.sha256(data).hexdigest(),
                "content": content,
            }
        )
        seen.add(relative)
    return selected


def _validate_ai_code_string_list(result, key):
    value = result.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise AIProviderError(f"AI integration field '{key}' must be a string list.")
    return [item.strip() for item in value if item.strip()]


def validate_ai_code_integration(result, source_files, project_root=None):
    """Validate provider output and bind every edit to an approved source hash."""
    if not isinstance(result, dict):
        raise AIProviderError("The provider returned an invalid integration object.")
    summary = result.get("analysis_summary")
    if not isinstance(summary, str) or not summary.strip():
        raise AIProviderError("The AI integration analysis summary is missing.")
    validated = {
        "analysis_summary": summary.strip(),
        "integration_plan": _validate_ai_code_string_list(result, "integration_plan"),
        "warnings": _validate_ai_code_string_list(result, "warnings"),
        "dependencies": _validate_ai_code_string_list(result, "dependencies"),
        "validation_steps": _validate_ai_code_string_list(result, "validation_steps"),
        "manual_steps": _validate_ai_code_string_list(result, "manual_steps"),
        "changes": [],
        "new_files": [],
    }
    source_by_path = {item["path"]: item for item in source_files}
    changes = result.get("changes")
    new_files = result.get("new_files")
    if not isinstance(changes, list) or not isinstance(new_files, list):
        raise AIProviderError("AI integration changes and new_files must be lists.")
    seen = set()
    for change in changes:
        if not isinstance(change, dict):
            raise AIProviderError("Each AI source change must be an object.")
        path = _normalise_project_relative_path(change.get("path"))
        if path in seen or path not in source_by_path:
            raise AIProviderError(
                f"AI attempted to change an unapproved or duplicate file: {path}"
            )
        expected_hash = source_by_path[path]["sha256"]
        if str(change.get("original_sha256", "")).lower() != expected_hash.lower():
            raise AIProviderError(f"AI source hash does not match {path}.")
        content = change.get("updated_content")
        explanation = change.get("explanation")
        if not isinstance(content, str) or "\x00" in content:
            raise AIProviderError(f"AI returned invalid text content for {path}.")
        if not isinstance(explanation, str) or not explanation.strip():
            raise AIProviderError(f"AI did not explain the change to {path}.")
        validated["changes"].append(
            {
                "path": path,
                "original_sha256": expected_hash,
                "updated_content": content,
                "explanation": explanation.strip(),
            }
        )
        seen.add(path)
    root = Path(project_root).resolve() if project_root else None
    for new_file in new_files:
        if not isinstance(new_file, dict):
            raise AIProviderError("Each AI new-file entry must be an object.")
        path = _normalise_project_relative_path(new_file.get("path"))
        reason = _ai_code_block_reason(path)
        if reason:
            raise AIProviderError(f"AI proposed unsafe new path {path}: {reason}.")
        if path in seen or (root is not None and (root / path).exists()):
            raise AIProviderError(f"AI proposed a duplicate or existing new path: {path}")
        content = new_file.get("content")
        purpose = new_file.get("purpose")
        if not isinstance(content, str) or "\x00" in content:
            raise AIProviderError(f"AI returned invalid new-file content for {path}.")
        if not isinstance(purpose, str) or not purpose.strip():
            raise AIProviderError(f"AI did not explain the purpose of {path}.")
        validated["new_files"].append(
            {"path": path, "content": content, "purpose": purpose.strip()}
        )
        seen.add(path)
    if not validated["changes"] and not validated["new_files"]:
        raise AIProviderError("The provider returned no source changes or new files.")
    return validated


def validate_ai_integrated_source_files(integration_result):
    """Perform non-executing syntax checks on generated Python and JSON."""
    errors = []
    passed = []
    file_records = []
    for item in integration_result.get("changes", []):
        file_records.append((item["path"], item["updated_content"]))
    for item in integration_result.get("new_files", []):
        file_records.append((item["path"], item["content"]))
    for path, content in file_records:
        suffix = Path(path).suffix.lower()
        try:
            if suffix == ".py":
                ast.parse(content, filename=path)
                passed.append(f"{path}: Python syntax passed")
            elif suffix == ".json":
                json.loads(content)
                passed.append(f"{path}: JSON syntax passed")
            else:
                passed.append(f"{path}: text/path safety passed")
        except (SyntaxError, json.JSONDecodeError) as exc:
            line = getattr(exc, "lineno", None)
            location = f" line {line}" if line else ""
            errors.append(f"{path}:{location} {exc}")
    return {"passed": passed, "errors": errors}


def build_ai_code_integration_diff(integration_result, source_files):
    source_by_path = {item["path"]: item["content"] for item in source_files}
    parts = []
    for item in integration_result.get("changes", []):
        path = item["path"]
        parts.extend(
            difflib.unified_diff(
                source_by_path[path].splitlines(keepends=True),
                item["updated_content"].splitlines(keepends=True),
                fromfile="original/" + path,
                tofile="integrated/" + path,
            )
        )
    for item in integration_result.get("new_files", []):
        path = item["path"]
        parts.extend(
            difflib.unified_diff(
                [],
                item["content"].splitlines(keepends=True),
                fromfile="/dev/null",
                tofile="integrated/" + path,
            )
        )
    return "".join(parts) or "No textual difference was produced."


def _complete_package_exclusion(path, relative_path):
    """Return why an existing project item must not enter the package."""
    path = Path(path)
    relative = Path(relative_path)
    if path.is_symlink():
        return "symbolic link excluded to prevent copying outside the project"
    lower_name = path.name.lower()
    if path.is_dir():
        if lower_name in {
            item.lower() for item in AI_COMPLETE_PACKAGE_EXCLUDED_DIRECTORIES
        }:
            return "dependency, cache, environment, metadata, or old evidence directory"
        return ""
    if (
        lower_name in {
            item.lower() for item in AI_COMPLETE_PACKAGE_SECRET_FILENAMES
        }
        or lower_name.startswith(".env.")
        or path.suffix.lower() in AI_COMPLETE_PACKAGE_SECRET_SUFFIXES
    ):
        return "credential or secret file"
    try:
        size = path.stat().st_size
    except OSError:
        return "unreadable file"
    if size <= 2_000_000:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            text = None
        if text is not None and _source_contains_probable_secret(text):
            return "probable embedded credential value"
    return ""


def copy_complete_ai_project(source_root, destination):
    """Copy a project without credentials/caches and return an audit record."""
    source_root = Path(source_root).resolve()
    destination = Path(destination).resolve()
    if not source_root.is_dir():
        raise ValueError("The selected application project is no longer available.")
    if source_root.parent == source_root:
        raise ValueError(
            "For a complete package, select the application project folder—not "
            "an entire drive or filesystem root."
        )
    try:
        destination.relative_to(source_root)
    except ValueError:
        pass
    else:
        raise ValueError(
            "Choose an export folder outside the original application project."
        )
    if destination.exists():
        raise FileExistsError(f"Package destination already exists: {destination}")

    included = []
    omitted = []

    def raise_walk_error(error):
        raise error

    destination.mkdir(parents=True)
    for current, directory_names, file_names in os.walk(
        source_root,
        topdown=True,
        followlinks=False,
        onerror=raise_walk_error,
    ):
        current_path = Path(current)
        current_relative = current_path.relative_to(source_root)
        kept_directories = []
        for name in sorted(directory_names):
            source_path = current_path / name
            relative = current_relative / name
            reason = _complete_package_exclusion(source_path, relative)
            if reason:
                omitted.append(
                    {"path": relative.as_posix() + "/", "reason": reason}
                )
            else:
                kept_directories.append(name)
        directory_names[:] = kept_directories
        target_directory = destination / current_relative
        target_directory.mkdir(parents=True, exist_ok=True)
        for name in sorted(file_names):
            source_path = current_path / name
            relative = current_relative / name
            reason = _complete_package_exclusion(source_path, relative)
            if reason:
                omitted.append({"path": relative.as_posix(), "reason": reason})
                continue
            output_path = destination / relative
            output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, output_path)
            included.append(relative.as_posix())
    return {"included": sorted(included), "omitted": omitted}


def build_complete_ai_package_inventory(
    package_root,
    updated_paths=None,
    new_paths=None,
):
    """Return a SHA-256 inventory for every current package file."""
    package_root = Path(package_root)
    updated_paths = set(updated_paths or [])
    new_paths = set(new_paths or [])
    records = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(package_root).as_posix()
        if relative == "NN_STUDIO_INTEGRATION/package_inventory.json":
            continue
        if relative in updated_paths:
            category = "ai_updated_source"
        elif relative in new_paths:
            category = "ai_new_source"
        elif relative.startswith("NN_STUDIO_INTEGRATION/model/"):
            category = "selected_model"
        elif relative.startswith("NN_STUDIO_INTEGRATION/"):
            category = "integration_evidence"
        else:
            category = "original_project_file"
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        records.append(
            {
                "path": relative,
                "category": category,
                "size_bytes": path.stat().st_size,
                "sha256": digest.hexdigest(),
            }
        )
    return records


def verify_complete_ai_package_zip(package_root, archive_path):
    """Verify that the ZIP contains every file from the completed folder."""
    package_root = Path(package_root)
    expected = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    with zipfile.ZipFile(archive_path, "r") as archive:
        actual = {name for name in archive.namelist() if not name.endswith("/")}
        bad_member = archive.testzip()
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if bad_member or missing or unexpected:
        details = []
        if bad_member:
            details.append(f"damaged ZIP member: {bad_member}")
        if missing:
            details.append("missing: " + ", ".join(missing[:10]))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected[:10]))
        raise ValueError("Complete package verification failed: " + "; ".join(details))
    return {"file_count": len(expected), "zip_verified": True}


def request_ai_code_integration(
    provider_name,
    api_key,
    model,
    base_url,
    timeout_seconds,
    source_files,
    model_contract,
    user_goal,
    transport=None,
    project_root=None,
):
    """Request one strict source change set from the selected provider."""
    if provider_name not in OFFICIAL_PROVIDER_SETTINGS:
        raise AIProviderError(
            "Connect OpenAI, DeepSeek, Claude, or an OpenAI-compatible API first."
        )
    if not str(user_goal or "").strip():
        raise AIProviderError("Describe how the model should work in the application.")
    key_value = str(api_key or "").strip()
    if not key_value:
        raise AIProviderError("An API key is required for AI code integration.")
    model_value = str(model or "").strip()
    if not model_value:
        raise AIProviderError("A provider model name is required.")
    base_value = _validate_provider_base_url(base_url)
    timeout_value = float(timeout_seconds)
    payload_context = {
        "user_goal": str(user_goal).strip(),
        "model_contract": model_contract,
        "approved_application_files": source_files,
    }
    user_prompt = (
        "Analyse the approved application files and integrate the model contract. "
        "Return the smallest complete change set.\n\n"
        + json.dumps(payload_context, ensure_ascii=False, separators=(",", ":"))
    )
    sender = transport or _post_json_request
    headers = {
        "Authorization": f"Bearer {key_value}",
        "Content-Type": "application/json",
    }
    if provider_name == AI_PROVIDER_OPENAI:
        endpoint = _provider_endpoint(base_value, "/v1/responses")
        payload = {
            "model": model_value,
            "input": [
                {"role": "system", "content": AI_CODE_INTEGRATION_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "nn_studio_code_integration",
                    "strict": True,
                    "schema": AI_CODE_INTEGRATION_SCHEMA,
                }
            },
            "max_output_tokens": 24000,
            "store": False,
        }
        raw_text = _extract_openai_response_text(
            sender(endpoint, payload, headers, timeout_value)
        )
    elif provider_name == AI_PROVIDER_ANTHROPIC:
        endpoint = _provider_endpoint(base_value, "/v1/messages")
        schema_prompt = (
            AI_CODE_INTEGRATION_SYSTEM_PROMPT
            + "\n\nReturn one JSON object only. Exact JSON Schema:\n"
            + json.dumps(AI_CODE_INTEGRATION_SCHEMA, separators=(",", ":"))
        )
        payload = {
            "model": model_value,
            "system": schema_prompt,
            "messages": [{"role": "user", "content": user_prompt}],
            "max_tokens": 24000,
            "temperature": 0,
        }
        raw_text = _extract_anthropic_response_text(
            sender(
                endpoint,
                payload,
                _anthropic_request_headers(key_value),
                timeout_value,
            )
        )
    else:
        endpoint = _provider_endpoint(base_value, "/chat/completions")
        schema_prompt = (
            AI_CODE_INTEGRATION_SYSTEM_PROMPT
            + "\n\nReturn JSON only. Exact JSON Schema:\n"
            + json.dumps(AI_CODE_INTEGRATION_SCHEMA, separators=(",", ":"))
        )
        payload = {
            "model": model_value,
            "messages": [
                {"role": "system", "content": schema_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": 24000,
            "stream": False,
        }
        raw_text = _extract_deepseek_response_text(
            sender(endpoint, payload, headers, timeout_value)
        )
    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise AIProviderError("The provider returned invalid integration JSON.") from exc
    return validate_ai_code_integration(
        parsed,
        source_files,
        project_root=project_root,
    )
