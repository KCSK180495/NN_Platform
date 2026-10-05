"""App for NN Training Studio."""

import queue
import tkinter as tk
from nn_training_studio.branding import (
    apply_window_branding,
    configure_windows_application_identity,
)
from nn_training_studio.constants import (
    AI_PROVIDER_LOCAL,
    APPLICATION_NAME,
    APP_VERSION,
    CUSTOM_FILTER_MODE_EXPERT,
    DATA_MODE_TABULAR,
    DEFAULT_CUSTOM_FILTER_CODE,
    DEFAULT_CUSTOM_MODEL_CODE,
    DEVICE_AUTO,
    FILTER_EXPORT_REPLACE,
    TASK_CLASSIFICATION,
    WORKSPACE_SIGNAL,
)
from nn_training_studio.settings import (
    _application_settings_path,
)
from nn_training_studio.ui.wizard_ai_settings import (
    AISettingsMixin,
)
from nn_training_studio.ui.wizard_ai_workflow import (
    AIWorkflowMixin,
)
from nn_training_studio.ui.wizard_custom_filters import (
    CustomFiltersMixin,
)
from nn_training_studio.ui.wizard_data_loading import (
    DataLoadingMixin,
)
from nn_training_studio.ui.wizard_feature_selection import (
    FeatureSelectionMixin,
)
from nn_training_studio.ui.wizard_filter_workspace import (
    FilterWorkspaceMixin,
)
from nn_training_studio.ui.wizard_guided import (
    GuidedWorkflowMixin,
)
from nn_training_studio.ui.wizard_home import (
    HomeMixin,
)
from nn_training_studio.ui.wizard_layout import (
    LayoutMixin,
)
from nn_training_studio.ui.wizard_model_settings import (
    ModelSettingsMixin,
)
from nn_training_studio.ui.wizard_navigation import (
    NavigationMixin,
)
from nn_training_studio.ui.wizard_preprocessing import (
    PreprocessingMixin,
)
from nn_training_studio.ui.wizard_projects_settings import (
    ProjectsSettingsMixin,
)
from nn_training_studio.ui.wizard_training_filters import (
    TrainingFiltersMixin,
)
from nn_training_studio.ui.wizard_training_workflow import (
    TrainingWorkflowMixin,
)


class NNWizardApp(
    GuidedWorkflowMixin,
    LayoutMixin,
    AISettingsMixin,
    HomeMixin,
    FilterWorkspaceMixin,
    ProjectsSettingsMixin,
    NavigationMixin,
    DataLoadingMixin,
    AIWorkflowMixin,
    TrainingFiltersMixin,
    CustomFiltersMixin,
    FeatureSelectionMixin,
    PreprocessingMixin,
    ModelSettingsMixin,
    TrainingWorkflowMixin,
    tk.Tk,
):
    def __init__(self):
        configure_windows_application_identity()
        super().__init__()

        self.branding_details = apply_window_branding(
            self,
            make_default=True,
        )
        self.title(f"{APPLICATION_NAME} {APP_VERSION}")
        self.geometry("1280x800")
        self.minsize(760, 520)

        self.ui_mode_var = tk.StringVar(value="Guided")
        self._active_ui_mode = "Guided"
        self.guided_stage = 0
        self.guided_detail_step = None
        self.project_goal_var = tk.StringVar(value="")
        self.guided_goal_var = tk.StringVar(value=next(iter(self.GUIDED_GOALS)))
        self.guided_data_order_var = tk.StringVar(value="Independent rows")
        self.project_draft_path = None
        self._editor_drafts = {}
        self._pending_preparation_steps = set()
        self._last_training_log = ""
        self.current_step = 0
        self.current_view = "home"
        self.settings_return_view = "home"
        self._responsive_after_id = None
        self._responsive_breakpoint = None
        self._responsive_width = None
        self._content_mousewheel_bound = False
        self._home_responsive_widgets = {}

        self.signal_step_titles = [
            "Step 1: Load Signal / Tabular Dataset",
            "Step 2: Optional Analysis and AI Recommendation",
            "Step 3: Built-in Signal Filter",
            "Step 4: Customize Signal Filter",
            "Step 5: Select Task and Inputs",
            "Step 6: Preprocess Data",
            "Step 7: Customize Neural Network",
            "Step 8: Train, Evaluate, and Save Model"
        ]
        self.image_step_titles = [
            "Step 1: Load Image Dataset",
            "Step 2: Inspect Classes and Dataset",
            "Step 3: Image Pipeline Overview",
            "Step 4: Image Transformation Safety",
            "Step 5: Confirm Image Classification Task",
            "Step 6: Resize, Split, and Augment Images",
            "Step 7: Build Image Model",
            "Step 8: Train, Evaluate, and Save Model",
        ]
        self.step_titles = list(self.signal_step_titles)
        self.active_workspace = WORKSPACE_SIGNAL

        self.data_mode_var = tk.StringVar(value=DATA_MODE_TABULAR)
        self.raw_df = None
        self.filtered_df = None
        self.df = None
        self.file_path = None
        self.image_directory = None
        self.image_records = None
        self.image_class_names = []
        self.dataset_profile = None
        self.dataset_analysis_report = ""
        self.sampling_frequency_var = tk.StringVar(value="")
        self.ai_recommendation = None
        self.ai_recommendation_report = ""
        self.ai_provider_summary = None
        self.ai_request_running = False
        self.ai_generation = None
        self.ai_generation_report = ""
        self.ai_generation_provider_summary = None
        self.safe_filter_generation = None
        self.safe_filter_recommendation = None
        self.safe_model_generation = None
        self.safe_model_recommendation = None

        # API keys remain session-only unless the user explicitly enables
        # secure OS credential storage. They are never included in reports,
        # training metadata, project files, logs, or saved model packages.
        self.ai_provider_var = tk.StringVar(value=AI_PROVIDER_LOCAL)
        self.ai_model_var = tk.StringVar(value="")
        self.ai_base_url_var = tk.StringVar(value="")
        self.ai_api_key_var = tk.StringVar(value="")
        self.ai_timeout_var = tk.StringVar(value="90")
        self.ai_goal_var = tk.StringVar(value="")
        self.remember_ai_key_var = tk.BooleanVar(value=False)
        self.show_ai_key_var = tk.BooleanVar(value=False)
        self.profile_only_var = tk.BooleanVar(value=True)
        self.allow_sample_rows_var = tk.BooleanVar(value=False)
        self.ai_connection_testing = False
        self.ai_connection_state = "offline"
        self.ai_connection_message = (
            "Offline mode — manual training remains available."
        )
        self.ai_last_connection_check = None
        self.ai_last_connection_details = "Not tested in this session."
        self.ai_settings_path = _application_settings_path()
        self.guide_auto_open_var = tk.BooleanVar(value=False)
        self.guide_use_ai_var = tk.BooleanVar(value=False)
        self.guide_history = []
        self.studio_guide_window = None
        self.annotation_workspace_window = None
        self.guide_startup_scheduled = False

        # Standalone Filter & Export workspace. Its state is intentionally
        # separate from the model-training wizard until the user explicitly
        # chooses "Continue to Signal Training".
        self.filter_tool_source_df = None
        self.filter_tool_result_df = None
        self.filter_tool_source_path = None
        self.filter_tool_method_var = tk.StringVar(value="No filter")
        self.filter_tool_moving_window_var = tk.StringVar(value="5")
        self.filter_tool_ema_span_var = tk.StringVar(value="10")
        self.filter_tool_median_window_var = tk.StringVar(value="5")
        self.filter_tool_kalman_q_var = tk.StringVar(value="0.00001")
        self.filter_tool_kalman_r_var = tk.StringVar(value="0.01")
        self.filter_tool_safe_enabled_var = tk.BooleanVar(value=False)
        self.filter_tool_safe_filter_spec = None
        self.filter_tool_ai_prompt_var = tk.StringVar(
            value=(
                "Design a conservative filter that reduces noise while "
                "preserving transients, phase, harmonics, peaks, and fault "
                "signatures."
            )
        )
        self.filter_tool_sampling_frequency_var = tk.StringVar(value="")
        self.filter_tool_ai_status = (
            "Select columns, configure an AI provider in Settings, and "
            "describe the required filter."
        )
        self.filter_tool_ai_report = ""
        self.filter_tool_ai_provider_summary = None
        self.filter_tool_custom_enabled_var = tk.BooleanVar(value=False)
        self.filter_tool_custom_code = DEFAULT_CUSTOM_FILTER_CODE
        self.filter_tool_plot_start_var = tk.StringVar(value="0")
        self.filter_tool_plot_end_var = tk.StringVar(value="")
        self.filter_tool_plot_max_points_var = tk.StringVar(value="5000")
        self.filter_tool_plot_mode_var = tk.StringVar(value="Overlay")
        self.filter_tool_x_axis_var = tk.StringVar(value="Sample index")
        self.filter_tool_export_mode_var = tk.StringVar(
            value=FILTER_EXPORT_REPLACE
        )
        self.filter_tool_suffix_var = tk.StringVar(value="_filtered")
        self.filter_tool_save_report_var = tk.BooleanVar(value=True)
        self.filter_tool_export_status_var = tk.StringVar(
            value="No filtered dataset saved yet."
        )

        self.feature_cols = []
        self.label_col = ""
        self.target_cols = []

        self.trained_model = None
        self.scaler = None
        self.label_encoder = None
        self.target_scaler = None
        self.metadata = None
        self.history = None
        self.best_checkpoint_path = None
        self.last_saved_model_package_path = None
        self.training_run_id = None
        self.confusion_mat = None
        self.class_names = None
        self.y_test_result = None
        self.y_pred_result = None
        self.result_target_names = []
        self.anomaly_scores = None

        self.ui_queue = queue.Queue()
        self.training_running = False
        self.training_thread = None
        self.training_error_text = ""
        self.training_preflight_status_var = tk.StringVar(
            value="Preflight will run before training starts."
        )

        # Step 2: Built-in filter / smoothing
        self.filter_method_var = tk.StringVar(value="No filter")
        self.moving_window_var = tk.StringVar(value="5")
        self.ema_span_var = tk.StringVar(value="10")
        self.median_window_var = tk.StringVar(value="5")
        self.kalman_q_var = tk.StringVar(value="0.00001")
        self.kalman_r_var = tk.StringVar(value="0.01")

        # Step 3: Optional custom Python filter
        self.custom_filter_enabled_var = tk.BooleanVar(value=False)
        self.custom_filter_mode_var = tk.StringVar(
            value=CUSTOM_FILTER_MODE_EXPERT
        )
        self.custom_filter_code = DEFAULT_CUSTOM_FILTER_CODE
        self.selected_custom_filter_cols = []
        self.custom_filtered_df = None
        self.safe_filter_spec = None
        self.custom_filter_ai_prompt_var = tk.StringVar(
            value=(
                "Design a conservative filter that removes noise while "
                "preserving transients, phase, harmonics, and fault signatures."
            )
        )
        self.custom_filter_ai_report = ""

        # Step 4: Task and column selection
        self.task_type_var = tk.StringVar(value=TASK_CLASSIFICATION)
        self.label_var = tk.StringVar()
        self.forecast_horizon_var = tk.StringVar(value="1")
        self.anomaly_percentile_var = tk.StringVar(value="95")

        # Step 4
        self.missing_var = tk.StringVar(value="Drop rows")
        self.scaler_var = tk.StringVar(value="MinMaxScaler")
        self.target_scaler_var = tk.StringVar(value="StandardScaler")
        self.test_size_var = tk.StringVar(value="30")

        # Step 4 basic
        self.model_type_var = tk.StringVar(value="CNN-LSTM")
        self.window_size_var = tk.StringVar(value="200")
        self.stride_var = tk.StringVar(value="50")
        self.epochs_var = tk.StringVar(value="30")
        self.batch_size_var = tk.StringVar(value="32")
        self.training_device_var = tk.StringVar(value=DEVICE_AUTO)
        self.training_device_status_var = tk.StringVar(value="")
        self.validation_split_var = tk.StringVar(value="20")
        self.early_stopping_enabled_var = tk.BooleanVar(value=True)
        self.early_stopping_patience_var = tk.StringVar(value="10")
        self.early_stopping_min_delta_var = tk.StringVar(value="0")
        self.early_stopping_warmup_var = tk.StringVar(value="0")
        self._early_stopping_panels = []
        self.image_height_var = tk.StringVar(value="224")
        self.image_width_var = tk.StringVar(value="224")
        self.image_color_mode_var = tk.StringVar(value="RGB")
        self.image_augmentation_var = tk.BooleanVar(value=True)
        self.image_pretrained_var = tk.BooleanVar(value=True)

        # Step 4 advanced
        self.hidden_activation_var = tk.StringVar(value="relu")
        self.dropout_var = tk.StringVar(value="0.3")
        self.output_mode_var = tk.StringVar(value="Auto")
        self.output_units_var = tk.StringVar(value="")
        self.output_activation_var = tk.StringVar(value="softmax")
        self.loss_var = tk.StringVar(value="sparse_categorical_crossentropy")
        self.optimizer_var = tk.StringVar(value="Adam")
        self.lr_var = tk.StringVar(value="0.001")

        # Custom Python model settings
        self.custom_model_input_mode_var = tk.StringVar(
            value="Window-based (3D)"
        )
        self.custom_model_code = DEFAULT_CUSTOM_MODEL_CODE
        self.safe_model_spec = None
        self.custom_model_ai_prompt_var = tk.StringVar(
            value=(
                "Design a compact neural network for the confirmed task. "
                "Balance accuracy, training speed, and overfitting risk."
            )
        )
        self.custom_model_ai_report = ""

        self.load_ai_settings()
        self.create_layout()
        self.show_home()
        self.after(100, self.process_ui_queue)
        if self.guide_auto_open_var.get():
            self.guide_startup_scheduled = True
            self.after(650, self.open_studio_guide)


def main():
    """Configure TensorFlow and start the desktop event loop."""
    from .devices import configure_tensorflow_memory_growth

    configure_tensorflow_memory_growth()
    app = NNWizardApp()
    app.mainloop()
