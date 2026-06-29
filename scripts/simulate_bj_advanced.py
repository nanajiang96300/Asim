#!/usr/bin/env python3
"""
Advanced BJ-Deep-Unfolding ONNXIM Simulation with Memory and Pipeline Analysis
Includes: data movement simulation, memory hierarchy, pipeline pipelining effects, 
bandwidth analysis, and energy estimation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import argparse


class AdvancedBJSimulator:
    """Advanced simulator with memory and pipeline modeling"""
    
    # MIMO Configuration
    M = 64  # Receive antennas
    K = 8   # Transmit symbols
    NUM_ITER = 12  # Chebyshev iterations
    
    # Data sizes (bytes)
    DTYPE_SIZE = 4  # Float32
    
    # MIMO matrix sizes (in bytes)
    B_SIZE = M * M * DTYPE_SIZE  # Preconditioner matrix
    Y_SIZE = M * K * DTYPE_SIZE  # Estimated signal matrix
    H_SIZE = M * K * DTYPE_SIZE  # Channel matrix
    
    # Memory Hierarchy
    L1_CAPACITY = 64 * 1024  # 64 KB
    L2_CAPACITY = 256 * 1024  # 256 KB
    L3_CAPACITY = 8 * 1024 * 1024  # 8 MB
    SRAM_CAPACITY = 32 * 1024  # 32 KB on-chip SRAM
    
    # Memory Access Latencies (cycles)
    L1_LATENCY = 2
    L2_LATENCY = 10
    L3_LATENCY = 50
    DRAM_LATENCY = 100
    SRAM_LATENCY = 1
    
    # Bandwidth (bytes per cycle)
    L1_BW = 32
    L2_BW = 32
    L3_BW = 32
    DRAM_BW = 64
    SRAM_BW = 128
    
    # Energy (pJ per operation)
    MATMUL_ENERGY = 100  # pJ per FMA
    ADD_ENERGY = 5
    MUL_ENERGY = 5
    SUB_ENERGY = 5
    L1_ACCESS_ENERGY = 2
    L2_ACCESS_ENERGY = 10
    L3_ACCESS_ENERGY = 50
    DRAM_ACCESS_ENERGY = 200
    
    def __init__(self, layer_csv: str):
        """Initialize advanced simulator"""
        self.layer_csv = layer_csv
        self.df_layers = pd.read_csv(layer_csv)
        self.execution_trace = []
        self.memory_access_trace = []
        self.energy_breakdown = {}
        
    def _estimate_operand_size(self, op_type: str) -> Tuple[int, int]:
        """Estimate input/output data sizes for operations"""
        if op_type == 'MatMul':
            # B @ Y: (M x M) @ (M x K) -> (M x K)
            input_bytes = self.B_SIZE + self.Y_SIZE
            output_bytes = self.Y_SIZE
        elif op_type in ['Add', 'Sub', 'Mul']:
            # Element-wise operations
            input_bytes = 2 * self.Y_SIZE  # Two operands
            output_bytes = self.Y_SIZE
        else:
            input_bytes = output_bytes = 0
            
        return input_bytes, output_bytes
    
    def _estimate_memory_access_time(self, op_type: str, is_input: bool) -> int:
        """Estimate memory access latency for operation"""
        if op_type == 'MatMul':
            # MatMul dominates memory - uses SRAM for reused weights
            return self.SRAM_LATENCY if is_input else self.SRAM_LATENCY
        else:
            # Vector operations primarily from registers/L1
            return self.L1_LATENCY
    
    def _estimate_execution_latency(self, op_type: str, data_size: int) -> int:
        """Estimate pure execution latency without memory"""
        # Based on data size and operation type
        if op_type == 'MatMul':
            # (M x M) @ (M x K) = 2*M*M*K MACs
            num_macs = 2 * self.M * self.M * self.K // (self.M * self.K)
            return max(1, int(num_macs / 4))  # Assume 4 MACs per cycle capability
        else:
            # Vector operations: one cycle per M*K elements
            num_elements = data_size // self.DTYPE_SIZE
            return max(1, num_elements // (32 // self.DTYPE_SIZE))  # 32-byte vector width
    
    def simulate_with_memory_model(self) -> Dict:
        """Run simulation with memory hierarchy model"""
        total_compute_cycles = 0
        total_memory_cycles = 0
        total_energy = 0  # pJ
        memory_access_count = {'L1': 0, 'L2': 0, 'L3': 0, 'DRAM': 0, 'SRAM': 0}
        energy_breakdown = {
            'compute': 0,
            'memory': 0,
            'total': 0
        }
        
        for _, row in self.df_layers.iterrows():
            op_type = row['onnx_op']
            cycles = int(row['compute_cycles'])
            
            total_compute_cycles += cycles
            
            # Estimate data movement
            input_size, output_size = self._estimate_operand_size(op_type)
            mem_latency_in = self._estimate_memory_access_time(op_type, True)
            mem_latency_out = self._estimate_memory_access_time(op_type, False)
            
            # Update memory access counters
            if op_type == 'MatMul':
                memory_access_count['SRAM'] += 1
            else:
                memory_access_count['L1'] += 1
            
            # Calculate energy
            if op_type == 'MatMul':
                macs = 2 * self.M * self.M * self.K
                exec_energy = (macs // 4) * self.MATMUL_ENERGY
            elif op_type == 'Add':
                exec_energy = (self.M * self.K) * self.ADD_ENERGY
            elif op_type == 'Mul':
                exec_energy = (self.M * self.K) * self.MUL_ENERGY
            elif op_type == 'Sub':
                exec_energy = (self.M * self.K) * self.SUB_ENERGY
            else:
                exec_energy = 0
            
            energy_breakdown['compute'] += exec_energy
            
            # Memory energy (estimated as fraction of compute energy)
            mem_energy = input_size // 1024 * 0.1 + output_size // 1024 * 0.05
            energy_breakdown['memory'] += mem_energy
        
        energy_breakdown['total'] = energy_breakdown['compute'] + energy_breakdown['memory']
        
        # Estimate total memory stall cycles (overlapped with compute)
        memory_stall_cycles = int(total_compute_cycles * 0.1)  # Assume 10% memory stalls
        total_cycles_with_memory = total_compute_cycles + memory_stall_cycles
        
        # Estimate throughput (symbols per microsecond)
        core_freq_mhz = 1000
        throughput = self.K / (total_compute_cycles / core_freq_mhz)
        
        return {
            'total_cycles': total_compute_cycles,
            'memory_stall_cycles': memory_stall_cycles,
            'total_cycles_with_memory': total_cycles_with_memory,
            'total_latency_us': total_compute_cycles / core_freq_mhz,
            'latency_with_memory_us': total_cycles_with_memory / core_freq_mhz,
            'memory_access_count': memory_access_count,
            'energy_breakdown': energy_breakdown,
            'throughput_symbols_per_us': throughput,
            'avg_energy_per_symbol_pj': energy_breakdown['total'] / self.K if self.K > 0 else 0,
        }
    
    def simulate_with_pipeline(self) -> Dict:
        """Simulate with pipeline overlap between iterations"""
        base_results = self.simulate_with_memory_model()
        
        # Pipeline overlap between iterations
        # Assume some overlap due to pipelining
        num_iterations = self.NUM_ITER
        base_latency = base_results['total_latency_us']
        
        # With perfect pipelining, latency would be base_latency / num_iterations
        # Realistically, assume 50% overlap
        pipeline_overlap_ratio = 0.5
        pipeline_latency = base_latency * num_iterations / (1 + (num_iterations - 1) * pipeline_overlap_ratio)
        
        return {
            **base_results,
            'pipeline_overlap_ratio': pipeline_overlap_ratio,
            'pipelined_latency_us': pipeline_latency,
            'pipeline_speedup': base_latency / pipeline_latency,
        }
    
    def print_detailed_analysis(self, results: Dict):
        """Print detailed simulation analysis"""
        print("=" * 80)
        print("ADVANCED BJ-Deep-Unfolding ONNXIM Simulation Analysis")
        print("=" * 80)
        
        print("\n[1] EXECUTION CYCLES")
        print("-" * 80)
        print(f"  Compute Cycles:           {results['total_cycles']:,}")
        print(f"  Estimated Memory Stalls:  {results['memory_stall_cycles']:,} cycles")
        print(f"  Total w/ Memory:          {results['total_cycles_with_memory']:,} cycles")
        
        print("\n[2] LATENCY BREAKDOWN")
        print("-" * 80)
        print(f"  Compute Latency:          {results['total_latency_us']:.3f} μs")
        print(f"  Latency w/ Memory:        {results['latency_with_memory_us']:.3f} μs")
        if 'pipelined_latency_us' in results:
            print(f"  Pipelined Latency:        {results['pipelined_latency_us']:.3f} μs")
            print(f"  Pipeline Speedup:         {results['pipeline_speedup']:.2f}x")
        
        print("\n[3] MEMORY ACCESS PATTERN")
        print("-" * 80)
        for level, count in results['memory_access_count'].items():
            print(f"  {level:8s} accesses:  {count:3d}")
        
        print("\n[4] ENERGY BREAKDOWN")
        print("-" * 80)
        energy = results['energy_breakdown']
        total = energy['total']
        print(f"  Compute Energy:           {energy['compute']:>8.0f} pJ ({energy['compute']/total*100:>5.1f}%)")
        print(f"  Memory Energy:            {energy['memory']:>8.0f} pJ ({energy['memory']/total*100:>5.1f}%)")
        print(f"  Total Energy:             {energy['total']:>8.0f} pJ")
        print(f"  Energy per Symbol:        {results['avg_energy_per_symbol_pj']:>8.1f} pJ")
        
        print("\n[5] PERFORMANCE METRICS")
        print("-" * 80)
        print(f"  Throughput:               {results['throughput_symbols_per_us']:.2f} symbols/μs")
        print(f"  Symbols per Detection:    {self.K}")
        print(f"  MIMO Config:              {self.M}x{self.K}")
        print(f"  Iterations:               {self.NUM_ITER}")
        
        print("\n" + "=" * 80)


def create_comparison_plot(config_64x8: Dict, config_256x32: Dict, output_file: str):
    """Create comparison plot between two configurations"""
    
    configs = ['64x8', '256x32']
    latencies = [config_64x8['total_latency_us'], config_256x32['total_latency_us']]
    energies = [config_64x8['energy_breakdown']['total'], config_256x32['energy_breakdown']['total']]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Latency comparison
    axes[0].bar(configs, latencies, color=['#1f77b4', '#ff7f0e'])
    axes[0].set_ylabel('Latency (μs)')
    axes[0].set_title('Latency Comparison')
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(latencies):
        axes[0].text(i, v + 0.05, f'{v:.3f}μs', ha='center')
    
    # Energy comparison
    axes[1].bar(configs, energies, color=['#2ca02c', '#d62728'])
    axes[1].set_ylabel('Energy (pJ)')
    axes[1].set_title('Total Energy Comparison')
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(energies):
        axes[1].text(i, v + 100, f'{v:.0f}pJ', ha='center')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Advanced BJ-Deep-Unfolding Simulator')
    parser.add_argument('--data-csv', type=str, 
                       default='traces/bj_deep_unfolding_64x8_layer_mapping_detailed.csv',
                       help='Layer mapping CSV file')
    parser.add_argument('--output', type=str, default='reports/bj_advanced_simulation',
                       help='Output path for results')
    args = parser.parse_args()
    
    # Run simulation
    sim = AdvancedBJSimulator(args.data_csv)
    results_memory = sim.simulate_with_memory_model()
    results_pipeline = sim.simulate_with_pipeline()
    
    # Print detailed analysis
    sim.print_detailed_analysis(results_pipeline)
    
    # Export results
    output_dir = Path(args.output).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_file = output_dir / f"{Path(args.output).stem}_results.json"
    with open(json_file, 'w') as f:
        # Convert to JSON-serializable format
        json_data = {k: v for k, v in results_pipeline.items() if k != 'memory_access_count'}
        json_data['memory_access_count'] = results_pipeline['memory_access_count']
        json.dump(json_data, f, indent=2)
    
    print(f"\nResults exported to: {json_file}")


if __name__ == '__main__':
    main()
