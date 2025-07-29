#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Attention Analysis Utilities for Style Injection Visualization

This module provides utilities to analyze and visualize attention maps
before softmax to understand the effect of style injection on attention patterns.
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
import os
from typing import List, Dict, Any, Optional
from custom_attention_processor4 import SemanticClipAttnProcessor

class AttentionAnalyzer:
    """
    Utility class for analyzing attention patterns in style injection models
    """
    
    def __init__(self, output_dir: str = "attention_analysis"):
        """
        Initialize the attention analyzer
        
        Args:
            output_dir: Directory to save analysis results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_attention_processors(self, model) -> List[SemanticClipAttnProcessor]:
        """
        Extract all SemanticClipAttnProcessor instances from a model
        
        Args:
            model: The model containing attention processors
            
        Returns:
            List of SemanticClipAttnProcessor instances
        """
        processors = []
        
        # Try to access through different model structures
        if hasattr(model, 'pipe') and hasattr(model.pipe, 'unet'):
            # For IPAdapterCustom structure
            unet = model.pipe.unet
        elif hasattr(model, 'unet'):
            # Direct UNet access
            unet = model.unet
        else:
            print("Warning: Could not find UNet in model structure")
            return processors
            
        # Extract processors
        for name, module in unet.named_modules():
            if isinstance(module, SemanticClipAttnProcessor):
                processors.append((name, module))
                
        print(f"Found {len(processors)} SemanticClipAttnProcessor instances")
        return processors
    
    def analyze_all_processors(self, processors: List[tuple], save_individual: bool = True) -> Dict[str, Any]:
        """
        Analyze attention patterns for all processors
        
        Args:
            processors: List of (name, processor) tuples
            save_individual: Whether to save individual processor analyses
            
        Returns:
            Combined analysis results
        """
        all_results = {
            'processor_results': {},
            'combined_stats': {
                'all_original_std': [],
                'all_style_injected_std': [],
                'all_std_ratios': [],
                'processor_names': []
            }
        }
        
        for name, processor in processors:
            print(f"Analyzing processor: {name}")
            
            # Get statistics for this processor
            stats = processor.compute_attention_std_before_softmax()
            
            if stats['original_std']:  # Only process if data exists
                all_results['processor_results'][name] = stats
                
                # Add to combined statistics
                all_results['combined_stats']['all_original_std'].extend(stats['original_std'])
                all_results['combined_stats']['all_style_injected_std'].extend(stats['style_injected_std'])
                all_results['combined_stats']['all_std_ratios'].extend(stats['std_ratio'])
                all_results['combined_stats']['processor_names'].extend([name] * len(stats['original_std']))
                
                # Save individual analysis if requested
                if save_individual:
                    safe_name = name.replace('.', '_').replace('/', '_')
                    save_path = os.path.join(self.output_dir, f"attention_analysis_{safe_name}.png")
                    processor.visualize_attention_std_comparison(save_path)
                    
        return all_results
    
    def create_combined_visualization(self, all_results: Dict[str, Any], save_path: Optional[str] = None):
        """
        Create a combined visualization of all attention processors
        
        Args:
            all_results: Results from analyze_all_processors
            save_path: Path to save the combined visualization
        """
        if save_path is None:
            save_path = os.path.join(self.output_dir, "combined_attention_analysis.png")
            
        combined_stats = all_results['combined_stats']
        
        if not combined_stats['all_original_std']:
            print("No data available for combined visualization")
            return
            
        # Create comprehensive visualization
        fig = plt.figure(figsize=(20, 12))
        
        # Plot 1: Overall comparison
        ax1 = plt.subplot(2, 3, 1)
        all_orig = combined_stats['all_original_std']
        all_style = combined_stats['all_style_injected_std']
        
        x_pos = np.arange(len(all_orig))
        ax1.plot(x_pos, all_orig, 'b-o', label='Original', alpha=0.7, markersize=3)
        ax1.plot(x_pos, all_style, 'r-s', label='Style-injected', alpha=0.7, markersize=3)
        ax1.set_xlabel('Layer Index (All Processors)')
        ax1.set_ylabel('Standard Deviation')
        ax1.set_title('Attention Std: Original vs Style-injected')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Ratio analysis
        ax2 = plt.subplot(2, 3, 2)
        ratios = combined_stats['all_std_ratios']
        ax2.plot(x_pos, ratios, 'g-^', alpha=0.7, markersize=3)
        ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Layer Index (All Processors)')
        ax2.set_ylabel('Std Ratio (Style/Original)')
        ax2.set_title('Standard Deviation Ratio')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Distribution comparison
        ax3 = plt.subplot(2, 3, 3)
        ax3.hist(all_orig, bins=30, alpha=0.7, label='Original', color='blue', density=True)
        ax3.hist(all_style, bins=30, alpha=0.7, label='Style-injected', color='red', density=True)
        ax3.set_xlabel('Standard Deviation')
        ax3.set_ylabel('Density')
        ax3.set_title('Distribution Comparison')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Per-processor summary
        ax4 = plt.subplot(2, 3, 4)
        processor_means_orig = []
        processor_means_style = []
        processor_labels = []
        
        for name, results in all_results['processor_results'].items():
            processor_means_orig.append(results['mean_original_std'])
            processor_means_style.append(results['mean_style_injected_std'])
            # Shorten processor names for display
            short_name = name.split('.')[-2] if '.' in name else name
            processor_labels.append(short_name)
        
        x_proc = np.arange(len(processor_labels))
        width = 0.35
        ax4.bar(x_proc - width/2, processor_means_orig, width, label='Original', alpha=0.7)
        ax4.bar(x_proc + width/2, processor_means_style, width, label='Style-injected', alpha=0.7)
        ax4.set_xlabel('Attention Processor')
        ax4.set_ylabel('Mean Standard Deviation')
        ax4.set_title('Per-Processor Mean Std')
        ax4.set_xticks(x_proc)
        ax4.set_xticklabels(processor_labels, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Ratio distribution
        ax5 = plt.subplot(2, 3, 5)
        ax5.hist(ratios, bins=30, alpha=0.7, color='green', density=True)
        ax5.axvline(x=1.0, color='k', linestyle='--', alpha=0.5, label='No change')
        ax5.axvline(x=np.mean(ratios), color='r', linestyle='-', alpha=0.7, label=f'Mean: {np.mean(ratios):.3f}')
        ax5.set_xlabel('Std Ratio (Style/Original)')
        ax5.set_ylabel('Density')
        ax5.set_title('Ratio Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Summary statistics
        ax6 = plt.subplot(2, 3, 6)
        ax6.axis('off')
        
        # Calculate summary statistics
        mean_orig = np.mean(all_orig)
        mean_style = np.mean(all_style)
        mean_ratio = np.mean(ratios)
        reduction_percentage = (1 - mean_ratio) * 100
        
        summary_text = f"""
        SUMMARY STATISTICS
        
        Overall Analysis:
        Mean Original Std: {mean_orig:.4f}
        Mean Style-injected Std: {mean_style:.4f}
        Mean Ratio: {mean_ratio:.4f}
        
        Effect Analysis:
        Std Reduction: {reduction_percentage:.1f}%
        Total Layers: {len(all_orig)}
        Processors: {len(all_results['processor_results'])}
        
        Interpretation:
        {'Style injection reduces attention spread' if mean_ratio < 1 else 'Style injection increases attention spread' if mean_ratio > 1 else 'Style injection has minimal effect'}
        
        Layers with reduced spread: {sum(1 for r in ratios if r < 1)} / {len(ratios)}
        Layers with increased spread: {sum(1 for r in ratios if r > 1)} / {len(ratios)}
        """
        
        ax6.text(0.05, 0.95, summary_text, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Combined attention analysis saved to: {save_path}")
        
    def clear_all_processors(self, processors: List[tuple]):
        """
        Clear attention data from all processors
        
        Args:
            processors: List of (name, processor) tuples
        """
        for name, processor in processors:
            processor.clear_attention_maps()
        print(f"Cleared attention data from {len(processors)} processors")

def analyze_model_attention(model, output_dir: str = "attention_analysis") -> Dict[str, Any]:
    """
    Convenience function to analyze attention patterns in a model
    
    Args:
        model: Model containing SemanticClipAttnProcessor instances
        output_dir: Directory to save results
        
    Returns:
        Analysis results
    """
    analyzer = AttentionAnalyzer(output_dir)
    
    # Extract processors
    processors = analyzer.extract_attention_processors(model)
    
    if not processors:
        print("No SemanticClipAttnProcessor instances found in model")
        return {}
    
    # Analyze all processors
    results = analyzer.analyze_all_processors(processors)
    
    # Create combined visualization
    analyzer.create_combined_visualization(results)
    
    # Clear processors for next run
    analyzer.clear_all_processors(processors)
    
    return results