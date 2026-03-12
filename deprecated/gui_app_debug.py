"""
Enhanced GUI with debug logging
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import traceback

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
    "Llama 4 Behemoth (405B)": LLAMA_4_BEHEMOTH,
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


# =============================================================================
# Debugging Helper
# =============================================================================

DEBUG = True  # Set to False to disable debug output

def debug_print(message):
    """Print debug messages to console"""
    if DEBUG:
        print(f"[DEBUG] {message}")


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
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight=2)
        self.root.rowconfigure(0, weight=1)
        
        # Initialize calculation type first
        self.calculation_type = tk.StringVar(value="TTFT")
        
        # Create main frames
        self.create_input_frame()
        self.create_output_frame()
        
        # Update calculation type UI
        self.update_calculation_type()
        
        debug_print("GUI initialized successfully")
    
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
        
        # Calculate Button
        calc_button = ttk.Button(input_frame, text="Calculate Performance",
                                command=self.calculate_performance,
                                style="Accent.TButton")
        calc_button.grid(row=row, column=0, pady=20, sticky=(tk.W, tk.E))
        
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
        memory_spin = ttk.Spinbox(section_frame, from_=1, to=1024,
                                 textvariable=self.memory_var, width=28)
        memory_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Compute Throughput (TFLOPS)
        ttk.Label(section_frame, text="Compute (TFLOPS):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.compute_var = tk.StringVar(value="1250")
        compute_spin = ttk.Spinbox(section_frame, from_=1, to=10000,
                                  textvariable=self.compute_var, width=28)
        compute_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Memory Bandwidth (GB/s)
        ttk.Label(section_frame, text="Memory BW (GB/s):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.memory_bw_var = tk.StringVar(value="8000")
        memory_bw_spin = ttk.Spinbox(section_frame, from_=1, to=10000,
                                    textvariable=self.memory_bw_var, width=28)
        memory_bw_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Network Bandwidth (GB/s)
        ttk.Label(section_frame, text="Network BW (GB/s):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.network_bw_var = tk.StringVar(value="900")
        network_bw_spin = ttk.Spinbox(section_frame, from_=1, to=10000,
                                     textvariable=self.network_bw_var, width=28)
        network_bw_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Kernel Launch Latency (µs)
        ttk.Label(section_frame, text="Kernel Launch Latency (µs):").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.kernel_latency_var = tk.StringVar(value="5.0")
        kernel_latency_spin = ttk.Spinbox(section_frame, from_=0.1, to=100, increment=0.1,
                                         textvariable=self.kernel_latency_var, width=28)
        kernel_latency_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # Parallelism Type
        ttk.Label(section_frame, text="Parallelism:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.parallelism_var = tk.StringVar()
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
        batch_size_spin = ttk.Spinbox(self.params_frame, from_=1, to=256,
                                     textvariable=self.batch_size_var, width=28)
        batch_size_spin.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=2)
        row += 1
        
        # TTFT-specific parameters
        self.ttft_params_row = row
        self.sequence_length_label = ttk.Label(self.params_frame, text="Sequence Length:")
        self.sequence_length_var = tk.StringVar(value="2048")
        self.sequence_length_spin = ttk.Spinbox(self.params_frame, from_=1, to=32768,
                                               textvariable=self.sequence_length_var, width=28)
        
        # Decode-specific parameters
        self.prefill_length_label = ttk.Label(self.params_frame, text="Prefill Length:")
        self.prefill_length_var = tk.StringVar(value="2048")
        self.prefill_length_spin = ttk.Spinbox(self.params_frame, from_=1, to=32768,
                                              textvariable=self.prefill_length_var, width=28)
        
        self.output_length_label = ttk.Label(self.params_frame, text="Output Length:")
        self.output_length_var = tk.StringVar(value="512")
        self.output_length_spin = ttk.Spinbox(self.params_frame, from_=1, to=8192,
                                             textvariable=self.output_length_var, width=28)
        
        return start_row + 1
    
    def create_output_frame(self):
        """Create right panel with results and visualization"""
        output_frame = ttk.Frame(self.root, padding="10")
        output_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(output_frame, text="Results", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, pady=(0, 10), sticky=tk.W)
        
        # Results text display
        results_frame = ttk.LabelFrame(output_frame, text="Performance Metrics", padding="10")
        results_frame.grid(row=1, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
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
        
        # Visualization canvas
        viz_frame = ttk.LabelFrame(output_frame, text="Resource Utilization", padding="10")
        viz_frame.grid(row=2, column=0, pady=5, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial empty plot
        self.ax.text(0.5, 0.5, 'Run calculation to see results', 
                    ha='center', va='center', fontsize=12, color='gray')
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()
    
    def update_model_info(self, *args):
        """Update model information display"""
        model_name = self.model_var.get()
        if model_name in MODEL_REGISTRY:
            model = MODEL_REGISTRY[model_name]
            info = f"{model.total_parameters/1e9:.1f}B params, {model.num_layers} layers, {model.attention_config.num_attention_heads} heads"
            self.model_info_label.config(text=info)
            debug_print(f"Model selected: {model_name}")
    
    def update_parallelism_options(self, *args):
        """Show/hide parallelism size based on parallelism type"""
        parallel_type = self.parallelism_var.get()
        if parallel_type in ["Data Parallel", "Tensor Parallel", "Pipeline Parallel", "3D Parallel"]:
            self.parallel_size_frame.grid()
        else:
            self.parallel_size_frame.grid_remove()
        debug_print(f"Parallelism type: {parallel_type}")
    
    def update_calculation_type(self):
        """Update parameter fields based on calculation type"""
        calc_type = self.calculation_type.get()
        debug_print(f"Calculation type: {calc_type}")
        
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
            
            debug_print(f"System params - Mem: {self.memory_var.get()}GB, "
                       f"Compute: {self.compute_var.get()}TF, "
                       f"MemBW: {self.memory_bw_var.get()}GB/s")
            
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
        debug_print("Calculate button clicked")
        
        try:
            # Get inputs
            model_name = self.model_var.get()
            if not model_name:
                error_msg = "Please select a model"
                print(f"\n❌ ERROR: {error_msg}")
                messagebox.showerror("Error", error_msg)
                return
            
            debug_print(f"Starting calculation for {model_name}")
            
            model = MODEL_REGISTRY[model_name]
            
            # Update kernel launch latency from GUI
            kernel_latency_us = float(self.kernel_latency_var.get())
            model.kernel_launch_latency = kernel_latency_us * 1e-6  # Convert µs to seconds
            debug_print(f"Kernel launch latency: {kernel_latency_us} µs")
            
            perf = InferencePerformance(model)
            
            gpu = self.get_system_constraints()
            debug_print("System constraints created")
            
            batch_size = int(self.batch_size_var.get())
            parallel_config = self.get_parallelism_config()
            
            calc_type = self.calculation_type.get()
            debug_print(f"Running {calc_type} calculation")
            
            # Run calculation
            if calc_type == "TTFT":
                sequence_length = int(self.sequence_length_var.get())
                debug_print(f"TTFT params: batch={batch_size}, seq_len={sequence_length}")
                
                kernel_latency_s = float(self.kernel_latency_var.get()) * 1e-6  # Convert µs to seconds
                debug_print(f"Kernel latency for calculation: {kernel_latency_s*1e6} µs")
                
                result = perf.calculate_achievable_ttft(
                    system_constraints=gpu,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    parallelism_config=parallel_config,
                    kernel_launch_latency=kernel_latency_s
                )
                
                debug_print(f"TTFT result: {result.achievable_ttft*1000:.2f}ms")
                self.display_ttft_results(result, model_name, batch_size, sequence_length, parallel_config)
                self.visualize_ttft_utilization(result)
            
            else:  # DECODE
                prefill_length = int(self.prefill_length_var.get())
                output_length = int(self.output_length_var.get())
                debug_print(f"Decode params: batch={batch_size}, prefill={prefill_length}, output={output_length}")
                
                kernel_latency_s = float(self.kernel_latency_var.get()) * 1e-6  # Convert µs to seconds
                debug_print(f"Kernel latency for calculation: {kernel_latency_s*1e6} µs")
                
                result = perf.calculate_decode_performance(
                    system_constraints=gpu,
                    batch_size=batch_size,
                    prefill_length=prefill_length,
                    output_length=output_length,
                    parallelism_config=parallel_config,
                    kernel_launch_latency=kernel_latency_s
                )
                
                debug_print(f"Decode result: {result.tokens_per_second_per_user:.2f} TPS")
                self.display_decode_results(result, model_name, batch_size, 
                                           prefill_length, output_length)
                self.visualize_decode_utilization(result)
            
            debug_print("Calculation completed successfully")
        
        except Exception as e:
            error_msg = f"An error occurred:\n{str(e)}"
            debug_print(f"ERROR: {error_msg}")
            debug_print(traceback.format_exc())
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
        self.ax.clear()
        
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
        bars = self.ax.bar(resources, utilizations, color=colors, alpha=0.7, edgecolor='black')
        
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
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{util:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Absolute value label above percentage
            self.ax.text(bar.get_x() + bar.get_width()/2., height + max(utilizations) * 0.08,
                        abs_val,
                        ha='center', va='bottom', fontsize=8, style='italic', color='#2c3e50')
        
        # Add 100% reference line
        self.ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
        self.ax.text(len(resources)-0.5, 102, '100% Capacity', 
                    fontsize=9, color='red', alpha=0.7)
        
        # Formatting
        self.ax.set_ylabel('Utilization (%)', fontsize=11, fontweight='bold')
        self.ax.set_title('System Resource Utilization', fontsize=12, fontweight='bold', pad=10)
        self.ax.set_ylim(0, max(max(utilizations) * 1.25, 110))  # Extra space for absolute labels
        self.ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight bottleneck
        bottleneck_idx = ['Compute', 'Memory Bandwidth', 'Network Bandwidth'].index(result.bottleneck_resource) if result.bottleneck_resource in ['Compute', 'Memory Bandwidth', 'Network Bandwidth'] else -1
        if bottleneck_idx >= 0:
            bars[bottleneck_idx].set_edgecolor('red')
            bars[bottleneck_idx].set_linewidth(3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def visualize_decode_utilization(self, result):
        """Create bar chart of resource utilization for decode"""
        self.ax.clear()
        
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
        bars = self.ax.bar(resources, utilizations, color=colors, alpha=0.7, edgecolor='black')
        
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
            self.ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{util:.1f}%',
                        ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Absolute value label above percentage
            self.ax.text(bar.get_x() + bar.get_width()/2., height + max(utilizations) * 0.08,
                        abs_val,
                        ha='center', va='bottom', fontsize=8, style='italic', color='#2c3e50')
        
        # Add 100% reference line
        self.ax.axhline(y=100, color='red', linestyle='--', linewidth=1, alpha=0.5)
        self.ax.text(len(resources)-0.5, 102, '100% Capacity', 
                    fontsize=9, color='red', alpha=0.7)
        
        # Formatting
        self.ax.set_ylabel('Average Utilization (%)', fontsize=11, fontweight='bold')
        self.ax.set_title('System Resource Utilization (Decode Phase)', 
                         fontsize=12, fontweight='bold', pad=10)
        self.ax.set_ylim(0, max(max(utilizations) * 1.25, 110))  # Extra space for absolute labels
        self.ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Highlight primary bottleneck
        bottleneck_idx = ['Compute', 'Memory Bandwidth', 'Network Bandwidth'].index(result.primary_bottleneck) if result.primary_bottleneck in ['Compute', 'Memory Bandwidth', 'Network Bandwidth'] else -1
        if bottleneck_idx >= 0:
            bars[bottleneck_idx].set_edgecolor('red')
            bars[bottleneck_idx].set_linewidth(3)
        
        self.fig.tight_layout()
        self.canvas.draw()


# =============================================================================
# Application Entry Point
# =============================================================================

def main():
    """Launch the GUI application"""
    debug_print("Starting LLM Inference Performance Analyzer GUI")
    root = tk.Tk()
    app = InferenceAnalyzerGUI(root)
    debug_print("Entering main loop")
    root.mainloop()


if __name__ == "__main__":
    main()
