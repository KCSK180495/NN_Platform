# NN Training Studio

A Python desktop application for data preparation, neural-network training,
evaluation, annotation, and deployment. Supports signal/tabular data, image
classification, optional YOLO object detection, and optional AI-provider assistance.

The original `UI_Meter_Ver_F_refined.py` application has been split into the
`nn_training_studio` package so each feature can be edited independently.

## Setup and launch

Use Python 3.11 or 3.12 in a virtual environment on a computer with a desktop
display and Tkinter. Python from python.org normally includes Tkinter on Windows
and macOS; Linux may require the distribution's `python3-tk` package.

```bash
python -m venv .venv
```

Activate the environment:

```powershell
# Windows PowerShell
.venv\Scripts\Activate.ps1
```

```bash
# macOS / Linux
source .venv/bin/activate
```

Then install the dependencies and launch:

```bash
python -m pip install -r requirements.txt
python main.py
```

You can also use `python -m nn_training_studio` or the original command
`python UI_Meter_Ver_F_refined.py`. The original filename is now a small launcher;
edit the package modules to change application behavior.

For object detection, install `requirements-detection.txt` instead. This adds
Ultralytics and its PyTorch dependencies. GPU support depends on your operating
system, drivers, and TensorFlow/PyTorch installation. CPU operation is supported.

AI-provider keys are optional and configured in the application's Settings.
Brand images were not part of the source upload. If available, place them in
`assets/` beside `main.py`; accepted filenames are in `constants.py`.

## Where to make changes

All paths below are inside `nn_training_studio/`.

| What you want to change | Files |
| --- | --- |
| Startup and main-window state | `app.py` |
| Labels, choices, defaults, JSON schemas, AI prompts | `constants.py` |
| Branding, icons, settings paths | `branding.py`, `settings.py` |
| Signal filters and safe templates | `filters.py`, `templates.py` |
| Dataset profiling and recommendations | `analysis.py` |
| Splitting, windows, scaling, missing values | `preprocessing.py` |
| Image-folder scanning and image datasets | `image_data.py` |
| Neural-network architectures and optimizers | `models.py` |
| Training callbacks, checkpoints, devices | `training.py`, `checkpoints.py`, `devices.py` |
| Loading models and running predictions | `model_io.py`, `inference.py` |
| Results exports and custom plots | `results.py`, `plotting.py` |
| AI requests, response validation, providers | `ai_transport.py`, `ai_validation.py`, `ai_providers.py` |
| Deployment and project integration | `deployment.py`, `provider_artifacts.py`, `integration.py` |
| YOLO datasets and result bundles | `detection.py` |
| Annotation data and AI assistance | `annotations.py`, `annotation_ai.py` |
| Studio guide responses | `studio_guide.py` |
| Separate application windows | `ui/deploy.py`, `ui/evaluation.py`, `ui/model_hub.py`, `ui/detection.py`, `ui/annotations.py`, `ui/guide.py`, `ui/results.py` |
| Main-window screens and interactions | `ui/wizard_*.py` (see below) |

The main application remains one Tk window. Its methods are grouped into mixin
classes: small classes that contribute related methods to `NNWizardApp`. These
mixins share the window's existing `self` state and are not standalone windows.

| Main-window feature | Module in `ui/` |
| --- | --- |
| Guided workflow and project drafts | `wizard_guided.py` |
| Layout and responsive navigation | `wizard_layout.py` |
| Provider credentials and settings state | `wizard_ai_settings.py` |
| Home page and starting projects | `wizard_home.py` |
| Standalone filter workspace | `wizard_filter_workspace.py` |
| Projects and settings pages | `wizard_projects_settings.py` |
| Advanced workflow navigation | `wizard_navigation.py` |
| CSV/image loading | `wizard_data_loading.py` |
| AI analysis and component generation | `wizard_ai_workflow.py` |
| Built-in filtering screen | `wizard_training_filters.py` |
| Custom filtering screen | `wizard_custom_filters.py` |
| Feature and target selection | `wizard_feature_selection.py` |
| Preprocessing controls | `wizard_preprocessing.py` |
| Model configuration | `wizard_model_settings.py` |
| Training workers, progress, plots, saving | `wizard_training_workflow.py` |

When adding a feature, keep reusable processing functions outside `ui/` and
import them explicitly into the screen that uses them. Import helpers from their
own module rather than from `app.py` to avoid circular imports.

## Validation

The headless test suite only needs the core dependencies:

```bash
python -m pip install -r requirements-core.txt
python -m unittest discover -s tests -v
python -m compileall -q nn_training_studio main.py UI_Meter_Ver_F_refined.py
```

Tests cover module wiring, mixin collisions, filtering, missing values,
forecast/group windows, image splits, annotations, and resource lookup.
This refactor preserves the uploaded function/method bodies, apart from asset
lookup adapting to the new package location and TensorFlow being imported only
when creating image datasets. TensorFlow device configuration runs at application
startup. It no longer runs merely from importing the package.

The full desktop UI, actual TensorFlow training, external AI requests, and YOLO
workflows still require testing on a configured desktop. Dependency ranges are
declared for setup; they are not a lockfile from a fully tested desktop environment.
