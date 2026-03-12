"""
LLM Inference Performance Analyzer - GUI Application

Modular GUI for analyzing LLM inference performance with:
- Model selection
- System configuration
- TTFT and Decode performance calculation
- Resource utilization visualization
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.ticker import LogLocator, FuncFormatter
import numpy as np

from speedometer import Speedometer
from llm_configs import (
    LLAMA_4_SCOUT, LLAMA_4_MAVERICK, LLAMA_4_BEHEMOTH,
    LLAMA_3_8B, LLAMA_3_70B, LLAMA_2_7B,
    DEEPSEEK_V3, DEEPSEEK_3_2,
    MISTRAL_7B, MIXTRAL_8X7B, GPT3_175B
)
from inference_performance import (
    InferencePerformance,
    SystemConstraints,
    ParallelismConfig,
    ParallelismType
)


# =============================================================================
# Model Registry - Easy to extend
# =============================================================================

MODEL_REGISTRY = {
    "Llama 4 Scout (109B)": LLAMA_4_SCOUT,
    "Llama 4 Maverick (400B)": LLAMA_4_MAVERICK,    
    "Llama 3 8B": LLAMA_3_8B,
    "Llama 3 70B": LLAMA_3_70B,
    "Llama 2 7B": LLAMA_2_7B,
    "DeepSeek V3": DEEPSEEK_V3,
    "DeepSeek 3.2": DEEPSEEK_3_2,
    "Mistral 7B": MISTRAL_7B,
    "Mixtral 8x7B": MIXTRAL_8X7B,
    "GPT-3 175B": GPT3_175B
}

GPU_SPECS = ["A100-40GB", "A100-80GB", "H100-80GB", "MI300X"]

# GPU Preset configurations for quick loading
GPU_PRESETS = {
    "A100-40GB": {
        "memory": 40,
        "compute": 312,
        "memory_bw": 1555,
        "network_bw": 600
    },
    "A100-80GB": {
        "memory": 80,
        "compute": 312,
        "memory_bw": 2039,
        "network_bw": 600
    },
    "H100-80GB": {
        "memory": 80,
        "compute": 1979,
        "memory_bw": 3352,
        "network_bw": 900
    },
    "MI300X": {
        "memory": 192,
        "compute": 1307,
        "memory_bw": 5300,
        "network_bw": 800
    },
    "Custom": {
        "memory": 80,
        "compute": 300,
        "memory_bw": 2000,
        "network_bw": 600
    }
}

PARALLELISM_TYPES = {
    "None": ParallelismType.NONE,
    "Data Parallel": ParallelismType.DATA_PARALLEL,
    "Tensor Parallel": ParallelismType.TENSOR_PARALLEL,
    "Pipeline Parallel": ParallelismType.PIPELINE_PARALLEL,
    "3D Parallel": ParallelismType.FULL_3D
}

# Default ranges for batch parameter sweeps
# Edit these values to change the default min/max for each parameter
BATCH_PARAMETER_DEFAULTS = {
    "Memory (GB)": {"min": 256, "max": 5084},
    "Compute (TFLOPS)": {"min": 1000, "max": 30000},
    "Memory BW (GB/s)": {"min": 1000, "max": 30000},
    "Network BW (GB/s)": {"min": 400, "max": 1600},
    "Kernel Launch Latency (µs)": {"min": 1, "max": 50},
    "Batch Size": {"min": 1, "max": 128},
    "Sequence Length": {"min": 128, "max": 8192},
    "Prefill Length": {"min": 128, "max": 8192},
    "Output Length": {"min": 128, "max": 16384}
}


# =============================================================================
# Main GUI Application
# =============================================================================

class InferenceAnalyzerGUI:
    """Main GUI application for LLM inference performance analysis"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("LLM Inference Performance Analyzer")
        self.root.geometry("1400x900")
        
        # Configure grid weight for responsiveness
        self.root.columnconfigure(0, weight=1, minsize=350)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize calculation type first
        self.calculation_type = tk.StringVar(value="TTFT")
        
        # Create main frames
        self.create_input_frame()
        self.create_output_frame()
        
        # Update calculation type UI
        self.update_calculation_type()
    
    def create_input_frame(self):
        """Create left panel with all input controls"""
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        input_frame.columnconfigure(0, weight=1)
        
        # Title
        title_label = ttk.Label(input_frame, text="Configuration", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Model Selection Section
        row = 1
        row = self.create_model_section(input_frame, row)
        
        # System Configuration Section
        row = self.create_system_section(input_frame, row)
        
        # Calculation Type Section
        row = self.create_calculation_type_section(input_frame, row)
        
        # Parameters Section (dynamic)
        row = self.create_parameters_section(input_frame, row)
        
        # Batch Analysis Section
        row = self.create_batch_section(input_frame, row)
        
        # Calculate Button - Hidden (auto-calculation is enabled)
        # calc_button = ttk.Button(input_frame, text="Calculate Performance",
        #                         command=self.calculate_performance,
        #                         style="Accent.TButton")
        # calc_button.grid(row=row, column=0, pady=20, sticky=(tk.W, tk.E))
        
        # Configure button style
        style = ttk.Style()
        style.configure("Accent.TButton", font=("Arial", 12, "bold"))
    
    def create_model_section(self, parent, start_row):
        """Create model selection section"""
        section_frame = ttk.LabelFrame(parent, text="Model Selection", padding="10")
        section_frame.grid(row=start_row, column=0, pady=5, sticky=(tk.W, tk.E))
        section_frame.columnconfigure(1, weight=1)
        
        # Model dropdown
        ttk.Label(section_frame, text="Model:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(section_frame, textvariable=self.model_var,
                                   values=list(MODEL_REGISTRY.keys()),
                                   state="readonly", width=30)
        model_combo.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        model_combo.set("Llama 3 8B")
        
        # Model info display
        self.model_info_label = ttk.Label(section_frame, text="", foreground="gray")
        self.model_info_label.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(5, 0))
        
        # Update model info when selection changes
        self.model_var.trace('w', self.update_model_info)
        self.model_var.trace('w', self.auto_calculate)
        self.update_model_info()
        
        return start_row + 1
    
    def create_system_section(self, parent, start_row):
        """Create system configuration section"""
        section_frame = ttk.LabelFrame(parent, text="System Configuration", padding="10")
        section_frame.grid(row=start_row, column=0, pady=5, sticky=(tk.W, tk.E))
        section_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Memory Capacity (GB)
        ttk.Label(section_frame, text="Memory (GB):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.memory_var = tk.StringVar(value="192")
        self.memory_var.trace('w', self.auto_calculate)
        memory_spin = ttk.Spinbox(section_frame, from_=1, to=1024,
                                 textvariable=self.memory_var, width=28)
        memory_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Compute Throughput (TFLOPS)
        ttk.Label(section_frame, text="Compute (TFLOPS):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.compute_var = tk.StringVar(value="1250")
        self.compute_var.trace('w', self.auto_calculate)
        compute_spin = ttk.Spinbox(section_frame, from_=1, to=10000,
                                  textvariable=self.compute_var, width=28)
        compute_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Memory Bandwidth (GB/s)
        ttk.Label(section_frame, text="Memory BW (GB/s):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.memory_bw_var = tk.StringVar(value="8000")
        self.memory_bw_var.trace('w', self.auto_calculate)
        memory_bw_spin = ttk.Spinbox(section_frame, from_=1, to=10000,
                                    textvariable=self.memory_bw_var, width=28)
        memory_bw_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Network Bandwidth (GB/s)
        ttk.Label(section_frame, text="Network BW (GB/s):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.network_bw_var = tk.StringVar(value="900")
        self.network_bw_var.trace('w', self.auto_calculate)
        network_bw_spin = ttk.Spinbox(section_frame, from_=1, to=10000,
                                     textvariable=self.network_bw_var, width=28)
        network_bw_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Kernel Launch Latency (µs)
        ttk.Label(section_frame, text="Kernel Launch Latency (µs):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.kernel_latency_var = tk.StringVar(value="5.0")
        self.kernel_latency_var.trace('w', self.auto_calculate)
        kernel_latency_spin = ttk.Spinbox(section_frame, from_=0.1, to=100, increment=0.1,
                                         textvariable=self.kernel_latency_var, width=28)
        kernel_latency_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Parallelism Type
        ttk.Label(section_frame, text="Parallelism:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.parallelism_var = tk.StringVar()
        self.parallelism_var.trace('w', self.auto_calculate)
        parallel_combo = ttk.Combobox(section_frame, textvariable=self.parallelism_var,
                                     values=list(PARALLELISM_TYPES.keys()),
                                     state="readonly", width=30)
        parallel_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        parallel_combo.set("None")
        parallel_combo.bind("<<ComboboxSelected>>", self.update_parallelism_options)
        row += 1
        
        # Parallelism Size (conditional)
        self.parallel_size_frame = ttk.Frame(section_frame)
        self.parallel_size_frame.grid(row=row, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=2)
        
        ttk.Label(self.parallel_size_frame, text="Parallel Size:").pack(side=tk.LEFT)
        self.parallel_size_var = tk.StringVar(value="2")
        self.parallel_size_var.trace('w', self.auto_calculate)
        parallel_size_combo = ttk.Combobox(self.parallel_size_frame, 
                                          textvariable=self.parallel_size_var,
                                          values=["2", "4", "8", "16"],
                                          state="readonly", width=10)
        parallel_size_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.parallel_size_frame.grid_remove()  # Hidden by default
        
        return start_row + 1
    
    def create_calculation_type_section(self, parent, start_row):
        """Create calculation type selection section"""
        section_frame = ttk.LabelFrame(parent, text="Analysis Type", padding="10")
        section_frame.grid(row=start_row, column=0, pady=5, sticky=(tk.W, tk.E))
        
        # Radio buttons for calculation type
        ttk.Radiobutton(section_frame, text="Calculate Achievable TTFT",
                       variable=self.calculation_type, value="TTFT",
                       command=self.update_calculation_type).grid(row=0, column=0, sticky=tk.W, pady=2)
        
        ttk.Radiobutton(section_frame, text="Calculate Decode Performance",
                       variable=self.calculation_type, value="DECODE",
                       command=self.update_calculation_type).grid(row=1, column=0, sticky=tk.W, pady=2)
        
        return start_row + 1
    
    def create_parameters_section(self, parent, start_row):
        """Create parameters section (dynamic based on calculation type)"""
        self.params_frame = ttk.LabelFrame(parent, text="Parameters", padding="10")
        self.params_frame.grid(row=start_row, column=0, pady=5, sticky=(tk.W, tk.E))
        self.params_frame.columnconfigure(1, weight=1)
        
        # Common parameters
        row = 0
        
        # Batch Size
        ttk.Label(self.params_frame, text="Batch Size:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.batch_size_var = tk.StringVar(value="1")
        self.batch_size_var.trace('w', self.auto_calculate)
        batch_size_spin = ttk.Spinbox(self.params_frame, from_=1, to=256,
                                     textvariable=self.batch_size_var, width=28)
        batch_size_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # TTFT-specific parameters
        self.ttft_params_row = row
        self.sequence_length_label = ttk.Label(self.params_frame, text="Sequence Length:")
        self.sequence_length_var = tk.StringVar(value="2048")
        self.sequence_length_var.trace('w', self.auto_calculate)
        self.sequence_length_spin = ttk.Spinbox(self.params_frame, from_=1, to=32768,
                                               textvariable=self.sequence_length_var, width=28)
        
        # Decode-specific parameters
        self.prefill_length_label = ttk.Label(self.params_frame, text="Prefill Length:")
        self.prefill_length_var = tk.StringVar(value="2048")
        self.prefill_length_var.trace('w', self.auto_calculate)
        self.prefill_length_spin = ttk.Spinbox(self.params_frame, from_=1, to=32768,
                                              textvariable=self.prefill_length_var, width=28)
        
        self.output_length_label = ttk.Label(self.params_frame, text="Output Length:")
        self.output_length_var = tk.StringVar(value="512")
        self.output_length_var.trace('w', self.auto_calculate)
        self.output_length_spin = ttk.Spinbox(self.params_frame, from_=1, to=8192,
                                             textvariable=self.output_length_var, width=28)
        
        return start_row + 1
    
    def create_batch_section(self, parent, start_row):
        """Create batch analysis section for parameter sweeps"""
        section_frame = ttk.LabelFrame(parent, text="Batch Analysis (Parameter Sweep)", padding="10")
        section_frame.grid(row=start_row, column=0, pady=5, sticky=(tk.W, tk.E))
        section_frame.columnconfigure(1, weight=1)
        
        row = 0
        
        # Parameter selection
        ttk.Label(section_frame, text="Parameter:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.batch_param_var = tk.StringVar()
        
        # Define sweepable parameters
        self.batch_params = [
            "Memory (GB)",
            "Compute (TFLOPS)",
            "Memory BW (GB/s)",
            "Network BW (GB/s)",
            "Kernel Launch Latency (µs)",
            "Batch Size",
            "Sequence Length",
            "Prefill Length",
            "Output Length"
        ]
        
        batch_param_combo = ttk.Combobox(section_frame, textvariable=self.batch_param_var,
                                         values=self.batch_params,
                                         state="readonly", width=30)
        batch_param_combo.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        batch_param_combo.set("Batch Size")
        
        # Bind parameter change to update min/max defaults
        self.batch_param_var.trace_add('write', self.update_batch_defaults)
        
        row += 1
        
        # Min value
        ttk.Label(section_frame, text="Min Value:").grid(row=row, column=0, sticky=tk.W, pady=2)
        default_min = BATCH_PARAMETER_DEFAULTS["Batch Size"]["min"]
        self.batch_min_var = tk.StringVar(value=str(default_min))
        batch_min_spin = ttk.Spinbox(section_frame, from_=0.1, to=100000,
                                     textvariable=self.batch_min_var, width=28)
        batch_min_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Max value
        ttk.Label(section_frame, text="Max Value:").grid(row=row, column=0, sticky=tk.W, pady=2)
        default_max = BATCH_PARAMETER_DEFAULTS["Batch Size"]["max"]
        self.batch_max_var = tk.StringVar(value=str(default_max))
        batch_max_spin = ttk.Spinbox(section_frame, from_=0.1, to=100000,
                                     textvariable=self.batch_max_var, width=28)
        batch_max_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Number of points
        ttk.Label(section_frame, text="Number of Points:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.batch_points_var = tk.StringVar(value="10")
        batch_points_spin = ttk.Spinbox(section_frame, from_=2, to=100,
                                        textvariable=self.batch_points_var, width=28)
        batch_points_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Run Batch button
        batch_button = ttk.Button(section_frame, text="Run Batch",
                                 command=self.run_batch_analysis)
        batch_button.grid(row=row, column=0, columnspan=2, pady=(10, 0), sticky=(tk.W, tk.E))
        
        return start_row + 1
    
    def create_output_frame(self):
        """Create right panel with results and visualization"""
        output_frame = ttk.Frame(self.root, padding="10")
        output_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=2)
        output_frame.columnconfigure(1, weight=1)
        output_frame.rowconfigure(1, weight=1)
        output_frame.rowconfigure(2, weight=1)
        
        # Title
        title_label = ttk.Label(output_frame, text="Results", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        
        # Results text display (top left)
        results_frame = ttk.LabelFrame(output_frame, text="Performance Metrics", padding="10")
        results_frame.grid(row=1, column=0, pady=5, padx=(0, 5), sticky=(tk.W, tk.E, tk.N, tk.S))
        results_frame.columnconfigure(0, weight=1)
        results_frame.rowconfigure(0, weight=1)
        
        # Scrolled text widget
        text_scroll = ttk.Scrollbar(results_frame, orient=tk.VERTICAL)
        self.results_text = tk.Text(results_frame, height=15, wrap=tk.WORD,
                                    yscrollcommand=text_scroll.set,
                                    font=("Courier", 10))
        text_scroll.config(command=self.results_text.yview)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        text_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Speedometer gauge (top right)
        speedometer_frame = ttk.LabelFrame(output_frame, text="Performance Gauge", padding="10")
        speedometer_frame.grid(row=1, column=1, pady=5, padx=(5, 0), sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create speedometer figure
        self.fig_speedometer = Figure(figsize=(4, 4), dpi=100)
        self.ax_speedometer = self.fig_speedometer.add_subplot(111)
        self.fig_speedometer.tight_layout(pad=1.0)
        
        self.canvas_speedometer = FigureCanvasTkAgg(self.fig_speedometer, master=speedometer_frame)
        self.canvas_speedometer.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initialize speedometer (will be updated based on calculation type)
        self.speedometer = Speedometer(self.ax_speedometer, metric_type='TTFT', bad_value=1000, good_value=50)
        self.speedometer.draw(500)  # Initial neutral value
        self.canvas_speedometer.draw()
        
        # Visualization canvas (bottom, with 3 subplots)
        viz_frame = ttk.LabelFrame(output_frame, text="Performance Analysis", padding="10")
        viz_frame.grid(row=2, column=0, columnspan=2, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure with three subplots side by side
        self.fig = Figure(figsize=(18, 5), dpi=100)
        self.ax_resource = self.fig.add_subplot(131)  # Left: resource utilization bar chart
        self.ax_batch = self.fig.add_subplot(132)     # Middle: batch analysis (TTFT/TPS)
        self.ax_batch_resources = self.fig.add_subplot(133)  # Right: resource usage over batch
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot for resource chart
        self.ax_resource.text(0.5, 0.5, 'Run calculation\nto see results', 
                    ha='center', va='center', fontsize=11, color='gray')
        self.ax_resource.set_xlim(0, 1)
        self.ax_resource.set_ylim(0, 1)
        self.ax_resource.axis('off')
        
        # Initial empty plot for batch analysis
        self.ax_batch.text(0.5, 0.5, 'Run batch analysis\nto see results', 
                    ha='center', va='center', fontsize=11, color='gray')
        self.ax_batch.set_xlim(0, 1)
        self.ax_batch.set_ylim(0, 1)
        self.ax_batch.axis('off')
        
        # Initial empty plot for batch resource usage
        self.ax_batch_resources.text(0.5, 0.5, 'Run batch analysis\nto see results', 
                    ha='center', va='center', fontsize=11, color='gray')
        self.ax_batch_resources.set_xlim(0, 1)
        self.ax_batch_resources.set_ylim(0, 1)
        self.ax_batch_resources.axis('off')
        
        self.canvas.draw()
    
    def update_model_info(self, *args):
        """Update model information display"""
        model_name = self.model_var.get()
        if model_name in MODEL_REGISTRY:
            model = MODEL_REGISTRY[model_name]
            info = f"{model.total_parameters/1e9:.1f}B params, {model.num_layers} layers, {model.attention_config.num_attention_heads} heads"
            self.model_info_label.config(text=info)
    
    def auto_calculate(self, *args):
        """Auto-calculate on input change"""
        # Use after_idle to debounce rapid changes
        if hasattr(self, '_auto_calc_id'):
            self.root.after_cancel(self._auto_calc_id)
        self._auto_calc_id = self.root.after(300, self.calculate_performance)
    
    def update_parallelism_options(self, *args):
        """Show/hide parallelism size based on parallelism type"""
        parallel_type = self.parallelism_var.get()
        if parallel_type in ["Data Parallel", "Tensor Parallel", "Pipeline Parallel", "3D Parallel"]:
            self.parallel_size_frame.grid()
        else:
            self.parallel_size_frame.grid_remove()
    
    def update_calculation_type(self):
        """Update parameter fields based on calculation type"""
        calc_type = self.calculation_type.get()
        
        # Clear existing dynamic parameters
        for widget in [self.sequence_length_label, self.sequence_length_spin,
                      self.prefill_length_label, self.prefill_length_spin,
                      self.output_length_label, self.output_length_spin]:
            widget.grid_remove()
        
        # Show appropriate parameters
        row = self.ttft_params_row
        if calc_type == "TTFT":
            self.sequence_length_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            self.sequence_length_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        else:  # DECODE
            self.prefill_length_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            self.prefill_length_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
            row += 1
            self.output_length_label.grid(row=row, column=0, sticky=tk.W, pady=2)
            self.output_length_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Trigger auto-calculation after changing type
        self.auto_calculate()
    
    def get_parallelism_config(self):
        """Build ParallelismConfig from UI inputs"""
        parallel_type_str = self.parallelism_var.get()
        parallel_type = PARALLELISM_TYPES[parallel_type_str]
        
        if parallel_type == ParallelismType.NONE:
            return ParallelismConfig()
        
        parallel_size = int(self.parallel_size_var.get())
        
        if parallel_type == ParallelismType.TENSOR_PARALLEL:
            return ParallelismConfig(
                parallelism_type=parallel_type,
                tensor_parallel_size=parallel_size
            )
        elif parallel_type == ParallelismType.PIPELINE_PARALLEL:
            return ParallelismConfig(
                parallelism_type=parallel_type,
                pipeline_parallel_size=parallel_size
            )
        elif parallel_type == ParallelismType.DATA_PARALLEL:
            return ParallelismConfig(
                parallelism_type=parallel_type,
                data_parallel_size=parallel_size
            )
        elif parallel_type == ParallelismType.FULL_3D:
            # Use equal distribution for simplicity
            size_per_dim = int(parallel_size ** (1/3)) or 2
            return ParallelismConfig(
                parallelism_type=parallel_type,
                data_parallel_size=size_per_dim,
                tensor_parallel_size=size_per_dim,
                pipeline_parallel_size=size_per_dim
            )
        
        return ParallelismConfig()
    
    def get_system_constraints(self):
        """Build SystemConstraints from UI inputs"""
        try:
            memory_capacity = float(self.memory_var.get()) * 1e9  # GB to bytes
            compute_throughput = float(self.compute_var.get()) * 1e12  # TFLOPS to FLOPS
            memory_bandwidth = float(self.memory_bw_var.get()) * 1e9  # GB/s to bytes/s
            network_bandwidth = float(self.network_bw_var.get()) * 1e9  # GB/s to bytes/s
            
            return SystemConstraints(
                memory_capacity=memory_capacity,
                memory_bandwidth=memory_bandwidth,
                compute_throughput=compute_throughput,
                network_bandwidth=network_bandwidth
            )
        except ValueError as e:
            raise ValueError(f"Invalid system parameter: {e}")
    
    def calculate_performance(self):
        """Run performance calculation based on inputs"""
        try:
            # Get inputs
            model_name = self.model_var.get()
            if not model_name:
                error_msg = "Please select a model"
                print(f"\n❌ ERROR: {error_msg}")
                messagebox.showerror("Error", error_msg)
                return
            
            model = MODEL_REGISTRY[model_name]
            
            # Update kernel launch latency from GUI
            kernel_latency_us = float(self.kernel_latency_var.get())
            model.kernel_launch_latency = kernel_latency_us * 1e-6  # Convert µs to seconds
            
            perf = InferencePerformance(model)
            
            gpu = self.get_system_constraints()
            
            batch_size = int(self.batch_size_var.get())
            parallel_config = self.get_parallelism_config()
            
            calc_type = self.calculation_type.get()
            
            # Run calculation
            if calc_type == "TTFT":
                sequence_length = int(self.sequence_length_var.get())
                kernel_latency_s = float(self.kernel_latency_var.get()) * 1e-6  # Convert µs to seconds
                result = perf.calculate_achievable_ttft(
                    system_constraints=gpu,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    parallelism_config=parallel_config,
                    kernel_launch_latency=kernel_latency_s
                )
                self.display_ttft_results(result, model_name, batch_size, sequence_length, parallel_config)
                self.visualize_ttft_utilization(result)
            
            else:  # DECODE
                prefill_length = int(self.prefill_length_var.get())
                output_length = int(self.output_length_var.get())
                kernel_latency_s = float(self.kernel_latency_var.get()) * 1e-6  # Convert µs to seconds
                result = perf.calculate_decode_performance(
                    system_constraints=gpu,
                    batch_size=batch_size,
                    prefill_length=prefill_length,
                    output_length=output_length,
                    parallelism_config=parallel_config,
                    kernel_launch_latency=kernel_latency_s
                )
                self.display_decode_results(result, model_name, batch_size, 
                                           prefill_length, output_length, parallel_config)
                self.visualize_decode_utilization(result)
        
        except Exception as e:
            import traceback
            error_msg = f"An error occurred:\n{str(e)}"
            print(f"\n❌ CALCULATION ERROR:")
            print(error_msg)
            print("\nFull traceback:")
            traceback.print_exc()
            messagebox.showerror("Calculation Error", error_msg)
    
    def display_ttft_results(self, result, model_name, batch_size, seq_len, parallel_config):
        """Display TTFT calculation results"""
        self.results_text.delete(1.0, tk.END)
        
        # Get system config for display
        memory_gb = self.memory_var.get()
        compute_tflops = self.compute_var.get()
        memory_bw_gbs = self.memory_bw_var.get()
        
        # Format parallelism info
        if parallel_config is None:
            parallel_info = "\n  Parallelism:     None (Single GPU)"
        else:
            num_gpus = parallel_config.total_gpus
            if num_gpus > 1:
                parallel_info = f"\n  Parallelism:     {parallel_config.parallelism_type.name} (N={num_gpus} GPUs)"
            else:
                parallel_info = "\n  Parallelism:     None (Single GPU)"
        
        output = f"""
{'='*60}
ACHIEVABLE TIME TO FIRST TOKEN (TTFT)
{'='*60}

Configuration:
  Model:           {model_name}{parallel_info}
  GPU Memory:      {memory_gb} GB (per GPU)
  GPU Compute:     {compute_tflops} TFLOPS (per GPU)
  Memory BW:       {memory_bw_gbs} GB/s (per GPU)
  Batch Size:      {batch_size}
  Sequence Length: {seq_len}

Performance Metrics:
  Achievable TTFT: {result.achievable_ttft*1000:.2f} ms
  Throughput:      {batch_size/result.achievable_ttft:.2f} requests/sec

Resource Utilization:
  Compute:         {result.compute_utilization*100:.1f}%
  Memory BW:       {result.memory_bandwidth_utilization*100:.1f}%
  Network BW:      {result.network_bandwidth_utilization*100:.1f}%
  Memory Usage:    {result.memory_utilization*100:.1f}%

Bottleneck Analysis:
  Primary:         {result.bottleneck_resource}
  {"⚠️  OUT OF MEMORY" if result.memory_utilization > 1.0 else "✓ Memory fits"}

Resource Details:
  Memory Used:     {result.memory_used/1e9:.2f} GB
  Memory Available:{result.memory_available/1e9:.2f} GB
  Compute Used:    {result.compute_used/1e12:.2f} TFLOP/s
  Compute Available:{result.compute_available/1e12:.2f} TFLOP/s
"""
        
        self.results_text.insert(1.0, output)
    
    def display_decode_results(self, result, model_name, batch_size, 
                               prefill_len, output_len, parallel_config):
        """Display decode calculation results"""
        self.results_text.delete(1.0, tk.END)
        
        # Get system config for display
        memory_gb = self.memory_var.get()
        compute_tflops = self.compute_var.get()
        memory_bw_gbs = self.memory_bw_var.get()
        
        # Format parallelism info
        if parallel_config is None:
            parallel_info = "\n  Parallelism:     None (Single GPU)"
        else:
            num_gpus = parallel_config.total_gpus
            if num_gpus > 1:
                parallel_info = f"\n  Parallelism:     {parallel_config.parallelism_type.name} (N={num_gpus} GPUs)"
            else:
                parallel_info = "\n  Parallelism:     None (Single GPU)"
        
        output = f"""
{'='*60}
DECODE PHASE PERFORMANCE
{'='*60}

Configuration:
  Model:           {model_name}{parallel_info}
  GPU Memory:      {memory_gb} GB (per GPU)
  GPU Compute:     {compute_tflops} TFLOPS (per GPU)
  Memory BW:       {memory_bw_gbs} GB/s (per GPU)
  Batch Size:      {batch_size}
  Prefill Length:  {prefill_len}
  Output Length:   {output_len}

Performance Metrics:
  Total Time:      {result.total_decode_time*1000:.2f} ms
  Avg Step Time:   {result.avg_step_time*1000:.3f} ms
  
  TPS (per user):  {result.tokens_per_second_per_user:.2f} tokens/sec
  Throughput:      {result.total_throughput:.2f} tokens/sec
  
  Time per Token:  {1000/result.tokens_per_second_per_user:.2f} ms/token

Resource Utilization (Average):
  Compute:         {result.avg_compute_utilization*100:.1f}%
  Memory BW:       {result.avg_memory_bw_utilization*100:.1f}%
  Network BW:      {result.avg_network_bw_utilization*100:.1f}%
  Memory Usage:    {result.avg_memory_utilization*100:.1f}%

Bottleneck Analysis:
  Primary:         {result.primary_bottleneck}
  Breakdown:       {', '.join(f'{k}: {v} steps' for k, v in result.bottleneck_breakdown.items())}

Efficiency:
  System Efficiency: {min(result.avg_compute_utilization, result.avg_memory_bw_utilization)*100:.1f}%
  {"⚠️  Memory Bandwidth Limited" if result.primary_bottleneck == "Memory Bandwidth" else "✓ Compute Limited" if result.primary_bottleneck == "Compute" else "✓ Network Limited"}
"""
        
        self.results_text.insert(1.0, output)
    
    def visualize_ttft_utilization(self, result):
        """Create bar chart of resource utilization for TTFT"""
        self.ax_resource.clear()
        
        # Prepare data
        resources = ['Compute', 'Memory\nBandwidth', 'Network\nBandwidth', 'Memory\nCapacity']
        utilizations = [
            result.compute_utilization * 100,
            result.memory_bandwidth_utilization * 100,
            result.network_bandwidth_utilization * 100,
            result.memory_utilization * 100
        ]
        
        # Color code: green (<80%), yellow (80-100%), red (>100%)
        colors = []
        for util in utilizations:
            if util > 100:
                colors.append('#e74c3c')  # Red
            elif util > 80:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#27ae60')  # Green
        
        # Create bar chart
        bars = self.ax_resource.bar(resources, utilizations, color=colors, alpha=0.7, edgecolor='black')
        
        # Prepare absolute values with units
        absolute_values = [
            f"{result.compute_used / 1e15:.2f} PFLOPs/s",  # Compute: PFLOPs/sec
            f"{result.memory_bandwidth_used / 1e12:.2f} TB/s",  # Memory BW: TB/sec
            f"{result.network_bandwidth_used / 1e12:.2f} TB/s",  # Network BW: TB/sec
            f"{result.memory_used / 1e12:.2f} TB"  # Memory: TB
        ]
        
        # Add value labels on bars (percentage + absolute value)
        for bar, util, abs_val in zip(bars, utilizations, absolute_values):
            height = bar.get_height()
            # Percentage label
            self.ax_resource.text(bar.get_x() + bar.get_width()/2., height,
                        f'{util:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Absolute value label above percentage
            self.ax_resource.text(bar.get_x() + bar.get_width()/2., height + max(utilizations) * 0.08,
                        abs_val,
                        ha='center', va='bottom', fontsize=8, style='italic', color='#2c3e50')
        
        # Add 100% reference line
        self.ax_resource.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_resource.text(len(resources)-0.5, 102, '100% Capacity', 
                    fontsize=9, color='red', alpha=0.7)
        
        # Formatting
        self.ax_resource.set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
        self.ax_resource.set_title('System Resource Utilization', fontsize=12, fontweight='bold', pad=10)
        self.ax_resource.set_ylim(0, max(max(utilizations) * 1.25, 110))  # Extra space for absolute labels
        self.ax_resource.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight bottleneck
        bottleneck_idx = ['Compute', 'Memory Bandwidth', 'Network Bandwidth'].index(result.bottleneck_resource) if result.bottleneck_resource in ['Compute', 'Memory Bandwidth', 'Network Bandwidth'] else -1
        if bottleneck_idx >= 0:
            bars[bottleneck_idx].set_edgecolor('red')
            bars[bottleneck_idx].set_linewidth(3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update speedometer with TTFT value (in milliseconds)
        ttft_ms = result.achievable_ttft * 1000
        self.speedometer = Speedometer(self.ax_speedometer, metric_type='TTFT', bad_value=1000, good_value=50)
        self.speedometer.draw(ttft_ms)
        self.canvas_speedometer.draw()
    
    def visualize_decode_utilization(self, result):
        """Create bar chart of resource utilization for decode"""
        self.ax_resource.clear()
        
        # Prepare data
        resources = ['Compute', 'Memory\nBandwidth', 'Network\nBandwidth', 'Memory\nCapacity']
        utilizations = [
            result.avg_compute_utilization * 100,
            result.avg_memory_bw_utilization * 100,
            result.avg_network_bw_utilization * 100,
            result.avg_memory_utilization * 100
        ]
        
        # Color code based on utilization
        colors = []
        for util in utilizations:
            if util > 100:
                colors.append('#e74c3c')  # Red
            elif util > 80:
                colors.append('#f39c12')  # Orange
            else:
                colors.append('#27ae60')  # Green
        
        # Create bar chart
        bars = self.ax_resource.bar(resources, utilizations, color=colors, alpha=0.7, edgecolor='black')
        
        # Prepare absolute values with units (average across decode steps)
        absolute_values = [
            f"{result.compute_throughput / 1e15:.2f} PFLOPs/s",  # Compute: PFLOPs/sec
            f"{result.memory_bandwidth / 1e12:.2f} TB/s",  # Memory BW: TB/sec
            f"{result.network_bandwidth / 1e12:.2f} TB/s",  # Network BW: TB/sec
            f"{result.memory_capacity / 1e12:.2f} TB"  # Memory: TB
        ]
        
        # Add value labels (percentage + absolute value)
        for bar, util, abs_val in zip(bars, utilizations, absolute_values):
            height = bar.get_height()
            # Percentage label
            self.ax_resource.text(bar.get_x() + bar.get_width()/2., height,
                        f'{util:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Absolute value label above percentage
            self.ax_resource.text(bar.get_x() + bar.get_width()/2., height + max(utilizations) * 0.08,
                        abs_val,
                        ha='center', va='bottom', fontsize=8, style='italic', color='#2c3e50')
        
        # Add 100% reference line
        self.ax_resource.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
        self.ax_resource.text(len(resources)-0.5, 102, '100% Capacity', 
                    fontsize=9, color='red', alpha=0.7)
        
        # Formatting
        self.ax_resource.set_ylabel('Average Utilization (%)', fontsize=11, fontweight='bold')
        self.ax_resource.set_title('System Resource Utilization (Decode Phase)', 
                         fontsize=12, fontweight='bold', pad=10)
        self.ax_resource.set_ylim(0, max(max(utilizations) * 1.25, 110))  # Extra space for absolute labels
        self.ax_resource.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight primary bottleneck
        bottleneck_idx = ['Compute', 'Memory Bandwidth', 'Network Bandwidth'].index(result.primary_bottleneck) if result.primary_bottleneck in ['Compute', 'Memory Bandwidth', 'Network Bandwidth'] else -1
        if bottleneck_idx >= 0:
            bars[bottleneck_idx].set_edgecolor('red')
            bars[bottleneck_idx].set_linewidth(3)
        
        self.fig.tight_layout()
        self.canvas.draw()
        
        # Update speedometer with TPS value (tokens per second)
        tps = result.tokens_per_second_per_user
        self.speedometer = Speedometer(self.ax_speedometer, metric_type='TPS', bad_value=10, good_value=200)
        self.speedometer.draw(tps)
        self.canvas_speedometer.draw()
    
    def update_batch_defaults(self, *args):
        """Update min/max defaults when batch parameter changes"""
        param_name = self.batch_param_var.get()
        if param_name in BATCH_PARAMETER_DEFAULTS:
            defaults = BATCH_PARAMETER_DEFAULTS[param_name]
            self.batch_min_var.set(str(defaults["min"]))
            self.batch_max_var.set(str(defaults["max"]))
    
    def run_batch_analysis(self):
        """Run batch analysis with parameter sweep"""
        try:
            # Get batch parameters
            param_name = self.batch_param_var.get()
            min_val = float(self.batch_min_var.get())
            max_val = float(self.batch_max_var.get())
            num_points = int(self.batch_points_var.get())
            
            if min_val >= max_val:
                messagebox.showerror("Error", "Min value must be less than max value")
                return
            
            if num_points < 2:
                messagebox.showerror("Error", "Number of points must be at least 2")
                return
            
            # Save original parameter value to restore after batch
            original_value = self._get_batch_parameter(param_name)
            
            # Generate sweep values with logarithmic spacing
            sweep_values = np.logspace(np.log10(min_val), np.log10(max_val), num_points)
            
            # Get current calculation type
            calc_type = self.calculation_type.get()
            
            # Run calculations for each value
            results_x = []
            results_y = []
            results_bandwidth = []  # TB/s
            results_compute = []     # PFLOPs/s
            
            for value in sweep_values:
                # Set the parameter
                self._set_batch_parameter(param_name, value)
                
                # Run calculation
                try:
                    result = self._run_single_calculation()
                    if result is not None:
                        results_x.append(value)
                        # Extract metric based on calculation type
                        if calc_type == "TTFT":
                            results_y.append(result.achievable_ttft * 1000)  # Convert to ms
                            # For TTFT, extract bandwidth and compute usage
                            results_bandwidth.append(result.memory_bandwidth_used / 1e12)  # Convert to TB/s
                            results_compute.append(result.compute_used / 1e15)  # Convert to PFLOPs/s
                        else:  # DECODE
                            results_y.append(result.tokens_per_second_per_user)
                            # For decode, use average values
                            results_bandwidth.append(result.memory_bandwidth / 1e12)  # Convert to TB/s
                            results_compute.append(result.compute_throughput / 1e15)  # Convert to PFLOPs/s
                except Exception as e:
                    print(f"Warning: Calculation failed at {param_name}={value}: {e}")
                    continue
            
            if len(results_x) == 0:
                messagebox.showerror("Error", "No successful calculations in batch run")
                # Restore original value even on error
                self._set_batch_parameter(param_name, original_value)
                return
            
            # Plot results
            self._plot_batch_results(param_name, results_x, results_y, calc_type)
            self._plot_batch_resources(param_name, results_x, results_bandwidth, results_compute)
            
            # Restore original parameter value
            self._set_batch_parameter(param_name, original_value)
            
        except Exception as e:
            import traceback
            error_msg = f"Batch analysis error:\n{str(e)}"
            print(f"\n❌ {error_msg}")
            print("\nFull traceback:")
            traceback.print_exc()
            messagebox.showerror("Batch Analysis Error", error_msg)
    
    def _set_batch_parameter(self, param_name, value):
        """Set a parameter value for batch analysis"""
        # Integer parameters - convert float to int
        integer_params = ["Batch Size", "Sequence Length", "Prefill Length", "Output Length"]
        
        if param_name in integer_params:
            value = int(round(value))
        
        param_map = {
            "Memory (GB)": self.memory_var,
            "Compute (TFLOPS)": self.compute_var,
            "Memory BW (GB/s)": self.memory_bw_var,
            "Network BW (GB/s)": self.network_bw_var,
            "Kernel Launch Latency (µs)": self.kernel_latency_var,
            "Batch Size": self.batch_size_var,
            "Sequence Length": self.sequence_length_var,
            "Prefill Length": self.prefill_length_var,
            "Output Length": self.output_length_var
        }
        
        if param_name in param_map:
            param_map[param_name].set(str(value))
    
    def _get_batch_parameter(self, param_name):
        """Get current parameter value for batch analysis"""
        param_map = {
            "Memory (GB)": self.memory_var,
            "Compute (TFLOPS)": self.compute_var,
            "Memory BW (GB/s)": self.memory_bw_var,
            "Network BW (GB/s)": self.network_bw_var,
            "Kernel Launch Latency (µs)": self.kernel_latency_var,
            "Batch Size": self.batch_size_var,
            "Sequence Length": self.sequence_length_var,
            "Prefill Length": self.prefill_length_var,
            "Output Length": self.output_length_var
        }
        
        if param_name in param_map:
            return float(param_map[param_name].get())
        return None
    
    def _run_single_calculation(self):
        """Run a single calculation and return result (for batch analysis)"""
        # Get inputs
        model_name = self.model_var.get()
        if not model_name:
            return None
        
        model = MODEL_REGISTRY[model_name]
        kernel_latency_us = float(self.kernel_latency_var.get())
        model.kernel_launch_latency = kernel_latency_us * 1e-6
        
        perf = InferencePerformance(model)
        gpu = self.get_system_constraints()
        batch_size = int(self.batch_size_var.get())
        parallel_config = self.get_parallelism_config()
        calc_type = self.calculation_type.get()
        
        if calc_type == "TTFT":
            sequence_length = int(self.sequence_length_var.get())
            kernel_latency_s = float(self.kernel_latency_var.get()) * 1e-6
            result = perf.calculate_achievable_ttft(
                system_constraints=gpu,
                batch_size=batch_size,
                sequence_length=sequence_length,
                parallelism_config=parallel_config,
                kernel_launch_latency=kernel_latency_s
            )
        else:  # DECODE
            prefill_length = int(self.prefill_length_var.get())
            output_length = int(self.output_length_var.get())
            kernel_latency_s = float(self.kernel_latency_var.get()) * 1e-6
            result = perf.calculate_decode_performance(
                system_constraints=gpu,
                batch_size=batch_size,
                prefill_length=prefill_length,
                output_length=output_length,
                parallelism_config=parallel_config,
                kernel_launch_latency=kernel_latency_s
            )
        
        return result
    
    def _plot_batch_results(self, param_name, x_values, y_values, calc_type):
        """Plot batch analysis results"""
        self.ax_batch.clear()
        
        # Determine y-axis label and title based on calculation type
        if calc_type == "TTFT":
            y_label = "TTFT (ms)"
            title = f"TTFT vs {param_name}"
        else:
            y_label = "Tokens per Second per User"
            title = f"TPS vs {param_name}"
        
        # Plot with logarithmic x-axis
        self.ax_batch.plot(x_values, y_values, 'o-', linewidth=2, markersize=6, color='#3498db')
        self.ax_batch.set_xscale('log')
        self.ax_batch.set_xlabel(param_name, fontsize=11, fontweight='bold')
        self.ax_batch.set_ylabel(y_label, fontsize=11, fontweight='bold')
        self.ax_batch.set_title(title, fontsize=12, fontweight='bold', pad=10)
        self.ax_batch.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Format x-axis to show actual numbers instead of powers of 10
        # Add more tick locations
        self.ax_batch.xaxis.set_major_locator(LogLocator(base=10.0, numticks=15))
        self.ax_batch.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=100))
        
        # Format labels as plain numbers
        def format_func(value, tick_number):
            if value >= 1000:
                return f'{int(value)}'
            elif value >= 1:
                return f'{value:.0f}'
            else:
                return f'{value:.2g}'
        
        self.ax_batch.xaxis.set_major_formatter(FuncFormatter(format_func))
        
        # Format axes
        self.ax_batch.tick_params(axis='both', which='major', labelsize=9)
        self.ax_batch.tick_params(axis='x', which='minor', labelsize=0)  # Hide minor tick labels
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def _plot_batch_resources(self, param_name, x_values, bandwidth_values, compute_values):
        """Plot batch resource usage (bandwidth and compute)"""
        self.ax_batch_resources.clear()
        
        # Create dual y-axis plot
        ax1 = self.ax_batch_resources
        ax2 = ax1.twinx()
        
        # Plot bandwidth on left axis
        line1 = ax1.plot(x_values, bandwidth_values, 'o-', linewidth=2, markersize=6, 
                        color='#e74c3c', label='Memory BW')
        ax1.set_xscale('log')
        ax1.set_xlabel(param_name, fontsize=10, fontweight='bold')
        ax1.set_ylabel('Memory Bandwidth (TB/s)', fontsize=10, fontweight='bold', color='#e74c3c')
        ax1.tick_params(axis='y', labelcolor='#e74c3c', labelsize=9)
        ax1.grid(True, alpha=0.3, linestyle='--', which='both')
        
        # Plot compute on right axis
        line2 = ax2.plot(x_values, compute_values, 's-', linewidth=2, markersize=6, 
                        color='#27ae60', label='Compute')
        ax2.set_ylabel('Compute (PFLOPs/s)', fontsize=10, fontweight='bold', color='#27ae60')
        ax2.tick_params(axis='y', labelcolor='#27ae60', labelsize=9)
        
        # Format x-axis
        ax1.xaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax1.xaxis.set_minor_locator(LogLocator(base=10.0, subs='auto', numticks=50))
        
        def format_func(value, tick_number):
            if value >= 1000:
                return f'{int(value)}'
            elif value >= 1:
                return f'{value:.0f}'
            else:
                return f'{value:.2g}'
        
        ax1.xaxis.set_major_formatter(FuncFormatter(format_func))
        ax1.tick_params(axis='x', which='major', labelsize=9)
        ax1.tick_params(axis='x', which='minor', labelsize=0)
        
        # Add title
        ax1.set_title(f'Resource Usage vs {param_name}', fontsize=11, fontweight='bold', pad=10)
        
        # Add legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=9)
        
        self.fig.tight_layout()
        self.canvas.draw()


# =============================================================================
# Application Entry Point
# =============================================================================

def main():
    """Launch the GUI application"""
    root = tk.Tk()
    app = InferenceAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
