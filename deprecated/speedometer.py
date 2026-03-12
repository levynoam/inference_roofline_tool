"""
Speedometer gauge visualization for performance metrics
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.lines import Line2D


class Speedometer:
    """
    A speedometer-style gauge that displays performance metrics.
    
    The needle moves from 7 o'clock position (bad) clockwise to 5 o'clock (good).
    Background color transitions from dark red (bad) -> dark green (mid) -> dark blue (good).
    """
    
    def __init__(self, ax, metric_type='TTFT', bad_value=1000, good_value=50):
        """
        Initialize speedometer.
        
        Args:
            ax: Matplotlib axes to draw on
            metric_type: 'TTFT' or 'TPS'
            bad_value: Value considered bad (7 o'clock position)
            good_value: Value considered good (5 o'clock position)
        """
        self.ax = ax
        self.metric_type = metric_type
        self.bad_value = bad_value
        self.good_value = good_value
        
        # For TTFT, lower is better. For TPS, higher is better
        self.lower_is_better = (metric_type == 'TTFT')
        
        # Angle range: 7 o'clock = -150° (210° from 3 o'clock), 5 o'clock = 30° (330° from 3 o'clock)
        self.start_angle = 210  # 7 o'clock position
        self.end_angle = -30    # 5 o'clock position
        self.angle_range = 240  # Total sweep
        
    def _value_to_angle(self, value):
        """Convert a value to an angle on the speedometer."""
        # Clamp value to range
        if self.lower_is_better:
            # For TTFT: lower is better
            clamped = np.clip(value, self.good_value, self.bad_value)
            # Normalize to 0-1 (0=good, 1=bad)
            normalized = (clamped - self.good_value) / (self.bad_value - self.good_value)
            # Invert for angle: good (low) should be at 5 o'clock (right), bad (high) at 7 o'clock (left)
            normalized = 1.0 - normalized
        else:
            # For TPS: higher is better
            clamped = np.clip(value, self.bad_value, self.good_value)
            # Normalize to 0-1 (0=bad, 1=good)
            normalized = (clamped - self.bad_value) / (self.good_value - self.bad_value)
        
        # Convert to angle (0=7 o'clock, 1=5 o'clock)
        angle = self.start_angle - normalized * self.angle_range
        return angle
    
    def _get_color_for_value(self, value):
        """Get background color based on value (dark red -> dark green -> dark blue)."""
        if self.lower_is_better:
            # For TTFT: lower is better
            clamped = np.clip(value, self.good_value, self.bad_value)
            normalized = (clamped - self.good_value) / (self.bad_value - self.good_value)
        else:
            # For TPS: higher is better
            clamped = np.clip(value, self.bad_value, self.good_value)
            normalized = (clamped - self.bad_value) / (self.good_value - self.bad_value)
        
        # Inverted: 0=good (blue), 0.5=mid (green), 1=bad (red)
        normalized = 1.0 - normalized
        
        if normalized < 0.5:
            # Bad to mid: red -> green
            t = normalized * 2  # 0 to 1
            r = 0.5 * (1 - t)   # 0.5 to 0
            g = 0.3 * t         # 0 to 0.3
            b = 0.0
        else:
            # Mid to good: green -> blue
            t = (normalized - 0.5) * 2  # 0 to 1
            r = 0.0
            g = 0.3 * (1 - t)   # 0.3 to 0
            b = 0.5 * t         # 0 to 0.5
        
        return (r, g, b)
    
    def draw(self, value):
        """
        Draw the speedometer with the given value.
        
        Args:
            value: Current metric value to display
        """
        self.ax.clear()
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        # Get angle and color for this value
        needle_angle = self._value_to_angle(value)
        bg_color = self._get_color_for_value(value)
        
        # Draw background arc with color gradient
        # Split into multiple segments for smoother gradient
        num_segments = 50
        angle_step = self.angle_range / num_segments
        
        for i in range(num_segments):
            segment_start = self.start_angle - i * angle_step
            segment_end = self.start_angle - (i + 1) * angle_step
            
            # Calculate color for this segment
            segment_normalized = i / num_segments
            if self.lower_is_better:
                segment_normalized = 1.0 - segment_normalized
            
            if segment_normalized < 0.5:
                t = segment_normalized * 2
                r = 0.5 * (1 - t)
                g = 0.3 * t
                b = 0.0
            else:
                t = (segment_normalized - 0.5) * 2
                r = 0.0
                g = 0.3 * (1 - t)
                b = 0.5 * t
            
            wedge = Wedge((0, -0.15), 1.0, segment_end, segment_start,
                         width=0.2, facecolor=(r, g, b), edgecolor='none')
            self.ax.add_patch(wedge)
        
        # Draw tick marks and labels
        self._draw_ticks()
        
        # Draw center circle
        center = Circle((0, -0.15), 0.05, facecolor='black', edgecolor='white', linewidth=2, zorder=10)
        self.ax.add_patch(center)
        
        # Draw needle
        needle_length = 0.75
        needle_angle_rad = np.radians(needle_angle)
        needle_x = needle_length * np.cos(needle_angle_rad)
        needle_y = needle_length * np.sin(needle_angle_rad) - 0.15
        
        needle = Line2D([0, needle_x], [-0.15, needle_y], 
                       linewidth=8, color='black', zorder=11)
        self.ax.add_line(needle)
        
        # Add value text in center
        if self.metric_type == 'TTFT':
            text = f'{value:.1f}\nms'
        else:
            text = f'{value:.1f}\ntok/s'
        
        self.ax.text(0, -0.45, text, ha='center', va='center',
                    fontsize=14, fontweight='bold', color='white', zorder=12)
        
        # Add title
        self.ax.text(0, 1.2, self.metric_type, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='black')
        
        # Set limits
        self.ax.set_xlim(-1.3, 1.3)
        self.ax.set_ylim(-1.3, 1.3)
    
    def _draw_ticks(self):
        """Draw tick marks and labels around the arc."""
        # Draw major ticks at start, mid, and end
        tick_angles = [self.start_angle, self.start_angle - self.angle_range/2, self.end_angle]
        
        if self.lower_is_better:
            tick_values = [self.bad_value, (self.bad_value + self.good_value)/2, self.good_value]
        else:
            tick_values = [self.bad_value, (self.bad_value + self.good_value)/2, self.good_value]
        
        for angle, value in zip(tick_angles, tick_values):
            angle_rad = np.radians(angle)
            
            # Outer tick
            x_outer = 1.0 * np.cos(angle_rad)
            y_outer = 1.0 * np.sin(angle_rad) - 0.15
            
            # Inner tick
            x_inner = 0.85 * np.cos(angle_rad)
            y_inner = 0.85 * np.sin(angle_rad) - 0.15
            
            # Draw tick
            tick = Line2D([x_inner, x_outer], [y_inner, y_outer],
                         linewidth=2, color='white', zorder=5)
            self.ax.add_line(tick)
            
            # Add label
            x_label = 1.15 * np.cos(angle_rad)
            y_label = 1.15 * np.sin(angle_rad) - 0.15
            
            if self.metric_type == 'TTFT':
                label_text = f'{int(value)}'
            else:
                label_text = f'{int(value)}'
            
            self.ax.text(x_label, y_label, label_text, ha='center', va='center',
                        fontsize=9, color='black', fontweight='bold')
