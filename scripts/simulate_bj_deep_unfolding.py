#!/usr/bin/env python3
"""
BJ-Deep-Unfolding ONNXIM Cycle Simulation
Simulates the cycle-level performance of BJ-deep-unfolding algorithm on the ONNX Inference Model simulator.

Algorithm: Block-Jacobi Preconditioned Chebyshev Iteration for MIMO Detection
Configurations: 64x8, 256x32
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from datetime import datetime


@dataclass
class OperationCycle:
    """Represents a single operation's cycle profile"""
    op_name: str
    op_type: str  # MatMul, Add, Sub, Mul
    layer_idx: int
    cycle_count: int
    data_in_latency: float = 0.0  # Memory latency for input
    data_out_latency: float = 0.0  # Memory latency for output


@dataclass
class LayerProfile:
    """Represents a single layer's cycle profile"""
    layer_idx: int
    operations: List[OperationCycle]
    total_cycles: int
    critical_path_cycles: int


class BJDeepUnfoldingSimulator:
    """Simulates BJ-Deep-Unfolding on ONNXIM"""
    
    # MIMO DETECTION MODEL PARAMETERS
    # BJ-Deep-Unfolding uses 12 Chebyshev iterations per detection symbol
    NUM_ITERATIONS = 12
    OPS_PER_ITERATION = 4  # BY, R, S, Y_update
    INIT_OPS = 1  # Y_0 initialization
    FINAL_OPS = 3  # Post-iteration MatMuls
    
    # ONNXIM Hardware Parameters (default config)
    SYSTOLIC_WIDTH = 128
    SYSTOLIC_HEIGHT = 128
    VECTOR_THROUGHPUT_BITS = 65536  # bits per cycle
    
    # Operation latencies (cycles)
    MATMUL_LATENCY = 1  # Systolic array latency
    VECTOR_OP_LATENCY = 1  # Add/Sub/Mul latency
    
    # Memory Access Patterns
    DRAM_ACCESS_LATENCY = 10  # cycles
    SRAM_ACCESS_LATENCY = 1   # cycles
    
    def __init__(self, config_path: str = None):
        """Initialize simulator with optional hardware config"""
        self.operations: List[OperationCycle] = []
        self.layers: List[LayerProfile] = []
        self.hardware_config = self._load_hardware_config(config_path)
        self.simulation_results = {}
        
    def _load_hardware_config(self, config_path: str = None) -> Dict:
        """Load hardware configuration from JSON file"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                return json.load(f)
        
        # Default hardware configuration
        return {
            'num_cores': 1,
            'core_freq_mhz': 1000,
            'systolic_width': self.SYSTOLIC_WIDTH,
            'systolic_height': self.SYSTOLIC_HEIGHT,
            'vector_throughput_bits': self.VECTOR_THROUGHPUT_BITS,
            'matmul_latency_cycles': self.MATMUL_LATENCY,
            'vector_op_latency_cycles': self.VECTOR_OP_LATENCY,
            'dram_latency_cycles': self.DRAM_ACCESS_LATENCY,
            'sram_latency_cycles': self.SRAM_ACCESS_LATENCY,
        }
    
    def load_layer_data(self, csv_path: str):
        """Load layer-wise cycle data from CSV file"""
        df = pd.read_csv(csv_path)
        
        current_layer_ops = []
        current_layer_idx = None
        
        for _, row in df.iterrows():
            layer_idx = int(row['layer_idx'])
            op_type = row['onnx_op']
            py_op = row['python_operation']
            cycles = int(row['compute_cycles'])
            
            op = OperationCycle(
                op_name=py_op,
                op_type=op_type,
                layer_idx=layer_idx,
                cycle_count=cycles,
                data_in_latency=self.DRAM_ACCESS_LATENCY if op_type == 'MatMul' else 0,
                data_out_latency=self.SRAM_ACCESS_LATENCY if op_type == 'MatMul' else 0,
            )
            
            self.operations.append(op)
            
            if layer_idx != current_layer_idx:
                if current_layer_ops:
                    self._create_layer_profile(current_layer_idx, current_layer_ops)
                current_layer_idx = layer_idx
                current_layer_ops = [op]
            else:
                current_layer_ops.append(op)
        
        # Process last layer
        if current_layer_ops:
            self._create_layer_profile(current_layer_idx, current_layer_ops)
    
    def _create_layer_profile(self, layer_idx: int, ops: List[OperationCycle]):
        """Create a layer profile from operations"""
        total_cycles = sum(op.cycle_count for op in ops)
        # Critical path is max cycle among operations (due to pipeline dependencies)
        critical_path = max(op.cycle_count for op in ops)
        
        layer = LayerProfile(
            layer_idx=layer_idx,
            operations=ops,
            total_cycles=total_cycles,
            critical_path_cycles=critical_path
        )
        self.layers.append(layer)
    
    def simulate(self) -> Dict:
        """Run simulation and compute cycle statistics"""
        if not self.layers:
            print("No operation data loaded. Call load_layer_data() first.")
            return {}
        
        total_cycles = 0
        total_matmul_cycles = 0
        total_vector_cycles = 0
        op_type_cycles = {}
        
        for layer in self.layers:
            for op in layer.operations:
                total_cycles += op.cycle_count
                
                if op.op_type == 'MatMul':
                    total_matmul_cycles += op.cycle_count
                else:
                    total_vector_cycles += op.cycle_count
                
                if op.op_type not in op_type_cycles:
                    op_type_cycles[op.op_type] = 0
                op_type_cycles[op.op_type] += op.cycle_count
        
        # Compute derived metrics
        num_layers = len(self.layers)
        avg_cycles_per_layer = total_cycles / num_layers if num_layers > 0 else 0
        num_operations = len(self.operations)
        avg_cycles_per_op = total_cycles / num_operations if num_operations > 0 else 0
        
        # Estimate latency in microseconds (assuming 1000 MHz clock)
        latency_us = total_cycles / self.hardware_config['core_freq_mhz']
        
        # Compute utilization metrics
        systolic_utilization = (total_matmul_cycles / total_cycles * 100) if total_cycles > 0 else 0
        vector_utilization = (total_vector_cycles / total_cycles * 100) if total_cycles > 0 else 0
        
        self.simulation_results = {
            'total_cycles': total_cycles,
            'num_layers': num_layers,
            'num_operations': num_operations,
            'avg_cycles_per_layer': avg_cycles_per_layer,
            'avg_cycles_per_op': avg_cycles_per_op,
            'latency_us': latency_us,
            'matmul_cycles': total_matmul_cycles,
            'vector_cycles': total_vector_cycles,
            'systolic_utilization_pct': systolic_utilization,
            'vector_utilization_pct': vector_utilization,
            'op_type_breakdown': op_type_cycles,
        }
        
        return self.simulation_results
    
    def print_summary(self):
        """Print simulation summary"""
        if not self.simulation_results:
            print("No simulation results. Run simulate() first.")
            return
        
        results = self.simulation_results
        print("=" * 70)
        print("BJ-Deep-Unfolding ONNXIM Cycle Simulation Results")
        print("=" * 70)
        print(f"\n  Total Cycles:           {results['total_cycles']:,}")
        print(f"  Total Latency:          {results['latency_us']:.3f} μs")
        print(f"  Num Layers:             {results['num_layers']}")
        print(f"  Num Operations:         {results['num_operations']}")
        print(f"  Avg Cycles/Layer:       {results['avg_cycles_per_layer']:.2f}")
        print(f"  Avg Cycles/Operation:   {results['avg_cycles_per_op']:.2f}")
        
        print(f"\n  Systolic Array Utilization:  {results['systolic_utilization_pct']:.2f}%")
        print(f"  Vector Unit Utilization:     {results['vector_utilization_pct']:.2f}%")
        
        print(f"\nCompute Breakdown:")
        print(f"  MatMul Cycles:          {results['matmul_cycles']:,} ({results['systolic_utilization_pct']:.2f}%)")
        print(f"  Vector Cycles (Add/Sub/Mul): {results['vector_cycles']:,} ({results['vector_utilization_pct']:.2f}%)")
        
        print(f"\nOperation Type Breakdown:")
        for op_type, cycles in sorted(results['op_type_breakdown'].items()):
            pct = (cycles / results['total_cycles'] * 100) if results['total_cycles'] > 0 else 0
            print(f"  {op_type:10s}: {cycles:5,} cycles ({pct:5.2f}%)")
        
        print("\n" + "=" * 70)
    
    def print_layer_details(self, max_layers: int = None):
        """Print per-layer cycle breakdown"""
        print("\n" + "=" * 70)
        print("Per-Layer Cycle Breakdown")
        print("=" * 70)
        print(f"{'Layer':<8} {'Operation':<20} {'Op Type':<10} {'Cycles':<10}")
        print("-" * 70)
        
        for layer in self.layers[:max_layers]:
            for i, op in enumerate(layer.operations):
                print(f"{layer.layer_idx:<8} {op.op_name:<20} {op.op_type:<10} {op.cycle_count:<10,}")
            
            total = layer.total_cycles
            print(f"{'':<8} {'LAYER TOTAL':<20} {'':<10} {total:<10,}")
            print("-" * 70)
    
    def export_results(self, output_path: str):
        """Export simulation results to JSON and CSV"""
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export JSON summary
        json_file = output_dir / f"{Path(output_path).stem}_summary.json"
        with open(json_file, 'w') as f:
            json.dump(self.simulation_results, f, indent=2)
        
        # Export detailed CSV
        csv_file = output_dir / f"{Path(output_path).stem}_detailed.csv"
        data = []
        for layer in self.layers:
            for op in layer.operations:
                data.append({
                    'layer_idx': op.layer_idx,
                    'operation': op.op_name,
                    'op_type': op.op_type,
                    'cycles': op.cycle_count,
                    'data_in_latency': op.data_in_latency,
                    'data_out_latency': op.data_out_latency,
                })
        
        df = pd.DataFrame(data)
        df.to_csv(csv_file, index=False)
        
        print(f"\nResults exported:")
        print(f"  JSON:  {json_file}")
        print(f"  CSV:   {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='BJ-Deep-Unfolding ONNXIM Cycle Simulator'
    )
    parser.add_argument(
        '--data-csv',
        type=str,
        default='traces/bj_deep_unfolding_64x8_layer_mapping_detailed.csv',
        help='Path to layer mapping CSV file'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to hardware configuration JSON file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='reports/bj_deep_unfolding_simulation',
        help='Output path for results'
    )
    parser.add_argument(
        '--print-layers',
        type=int,
        default=None,
        help='Number of layers to print details for (None = all)'
    )
    
    args = parser.parse_args()
    
    print(f"Starting BJ-Deep-Unfolding Simulation...")
    print(f"Data Source: {args.data_csv}")
    print(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Initialize simulator
    simulator = BJDeepUnfoldingSimulator(config_path=args.config)
    
    # Load data and run simulation
    simulator.load_layer_data(args.data_csv)
    simulator.simulate()
    
    # Print results
    simulator.print_summary()
    if args.print_layers:
        simulator.print_layer_details(max_layers=args.print_layers)
    elif args.print_layers is None:
        simulator.print_layer_details()
    
    # Export results
    simulator.export_results(args.output)


if __name__ == '__main__':
    main()
