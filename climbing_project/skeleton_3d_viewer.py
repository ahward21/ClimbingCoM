"""
3D Skeleton Climbing Visualization Tool
GUI application - double-click to run and select your JSON file.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import sys
import os


class SkeletonViewer:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("3D Skeleton Climbing Viewer")
        self.root.geometry("500x400")
        self.root.configure(bg='#1a1a2e')
        
        # Variables
        self.json_path = tk.StringVar()
        self.fps = tk.IntVar(value=30)
        self.rotate_camera = tk.BooleanVar(value=True)
        self.save_output = tk.BooleanVar(value=False)
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(
            self.root, 
            text="🧗 3D Skeleton Viewer", 
            font=('Segoe UI', 20, 'bold'),
            fg='#00d4ff',
            bg='#1a1a2e'
        )
        title.pack(pady=20)
        
        # File selection frame
        file_frame = tk.Frame(self.root, bg='#1a1a2e')
        file_frame.pack(pady=10, padx=20, fill='x')
        
        tk.Label(
            file_frame, 
            text="JSON File:", 
            font=('Segoe UI', 11),
            fg='white',
            bg='#1a1a2e'
        ).pack(anchor='w')
        
        entry_frame = tk.Frame(file_frame, bg='#1a1a2e')
        entry_frame.pack(fill='x', pady=5)
        
        self.file_entry = tk.Entry(
            entry_frame, 
            textvariable=self.json_path,
            font=('Segoe UI', 10),
            bg='#2d2d44',
            fg='white',
            insertbackground='white',
            relief='flat'
        )
        self.file_entry.pack(side='left', fill='x', expand=True, ipady=8)
        
        browse_btn = tk.Button(
            entry_frame,
            text="Browse...",
            command=self.browse_file,
            font=('Segoe UI', 10),
            bg='#00d4ff',
            fg='#1a1a2e',
            relief='flat',
            cursor='hand2'
        )
        browse_btn.pack(side='right', padx=(10, 0), ipady=5, ipadx=10)
        
        # Options frame
        options_frame = tk.Frame(self.root, bg='#1a1a2e')
        options_frame.pack(pady=20, padx=20, fill='x')
        
        # FPS setting
        fps_frame = tk.Frame(options_frame, bg='#1a1a2e')
        fps_frame.pack(fill='x', pady=5)
        
        tk.Label(
            fps_frame, 
            text="FPS:", 
            font=('Segoe UI', 11),
            fg='white',
            bg='#1a1a2e'
        ).pack(side='left')
        
        fps_spin = tk.Spinbox(
            fps_frame,
            from_=1,
            to=120,
            textvariable=self.fps,
            width=5,
            font=('Segoe UI', 10),
            bg='#2d2d44',
            fg='white',
            relief='flat'
        )
        fps_spin.pack(side='left', padx=10)
        
        # Checkboxes
        tk.Checkbutton(
            options_frame,
            text="Rotate camera during playback",
            variable=self.rotate_camera,
            font=('Segoe UI', 10),
            fg='white',
            bg='#1a1a2e',
            selectcolor='#2d2d44',
            activebackground='#1a1a2e',
            activeforeground='white'
        ).pack(anchor='w', pady=5)
        
        tk.Checkbutton(
            options_frame,
            text="Save as MP4 (requires FFmpeg)",
            variable=self.save_output,
            font=('Segoe UI', 10),
            fg='white',
            bg='#1a1a2e',
            selectcolor='#2d2d44',
            activebackground='#1a1a2e',
            activeforeground='white'
        ).pack(anchor='w', pady=5)
        
        # Buttons frame
        btn_frame = tk.Frame(self.root, bg='#1a1a2e')
        btn_frame.pack(pady=30)
        
        view_btn = tk.Button(
            btn_frame,
            text="▶  View Animation",
            command=self.start_visualization,
            font=('Segoe UI', 12, 'bold'),
            bg='#00d4ff',
            fg='#1a1a2e',
            relief='flat',
            cursor='hand2',
            width=20
        )
        view_btn.pack(ipady=10)
        
        # Status label
        self.status_label = tk.Label(
            self.root,
            text="Select a JSON file to begin",
            font=('Segoe UI', 9),
            fg='#888899',
            bg='#1a1a2e'
        )
        self.status_label.pack(pady=10)
        
    def browse_file(self):
        filepath = filedialog.askopenfilename(
            title="Select Skeleton JSON File",
            filetypes=[
                ("JSON files", "*.json"),
                ("All files", "*.*")
            ]
        )
        if filepath:
            self.json_path.set(filepath)
            self.status_label.config(text=f"Ready to visualize", fg='#00d4ff')
    
    def start_visualization(self):
        json_file = self.json_path.get()
        
        if not json_file:
            messagebox.showerror("Error", "Please select a JSON file first!")
            return
        
        if not os.path.exists(json_file):
            messagebox.showerror("Error", f"File not found: {json_file}")
            return
        
        try:
            self.status_label.config(text="Loading data...", fg='#ffcc00')
            self.root.update()
            
            joint_names, connections, frames = load_skeleton_data(json_file)
            
            output_path = None
            if self.save_output.get():
                output_path = filedialog.asksaveasfilename(
                    title="Save Animation As",
                    defaultextension=".mp4",
                    filetypes=[
                        ("MP4 video", "*.mp4"),
                        ("GIF animation", "*.gif")
                    ]
                )
                if not output_path:
                    self.save_output.set(False)
            
            self.status_label.config(text="Rendering animation...", fg='#00d4ff')
            self.root.update()
            
            # Hide the main window while showing animation
            self.root.withdraw()
            
            create_animation(
                joint_names,
                connections,
                frames,
                output_path=output_path,
                fps=self.fps.get(),
                rotate_camera=self.rotate_camera.get()
            )
            
            # Show main window again
            self.root.deiconify()
            self.status_label.config(text="Done! Select another file or close.", fg='#00ff88')
            
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON format:\n{e}")
            self.status_label.config(text="Error loading file", fg='#ff6b6b')
        except KeyError as e:
            messagebox.showerror("Data Error", f"Missing required key: {e}\n\nExpected format:\n- joint_names: [...]\n- skeleton_connections: [...]\n- frames: [...]")
            self.status_label.config(text="Error: Invalid data format", fg='#ff6b6b')
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred:\n{e}")
            self.status_label.config(text="Error occurred", fg='#ff6b6b')
        finally:
            self.root.deiconify()
    
    def run(self):
        self.root.mainloop()


def load_skeleton_data(json_path):
    """Load skeleton data from JSON file."""
    print(f"Loading data from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    joint_names = data['joint_names']
    connections = data['skeleton_connections']
    frames = data['frames']
    
    print(f"Loaded {len(frames)} frames with {len(joint_names)} joints")
    return joint_names, connections, frames


def calculate_data_bounds(frames):
    """Calculate the bounds of all skeleton data across all frames."""
    all_x, all_y, all_z = [], [], []
    
    for frame in frames:
        joints = frame['joints']
        for joint_data in joints.values():
            all_x.append(joint_data['x'])
            all_y.append(joint_data['y'])
            all_z.append(joint_data['z'])
    
    # Add padding
    padding = 0.1
    x_range = (min(all_x) - padding, max(all_x) + padding)
    y_range = (min(all_y) - padding, max(all_y) + padding)
    z_range = (min(all_z) - padding, max(all_z) + padding)
    
    # Make ranges equal for proper aspect ratio
    max_range = max(
        x_range[1] - x_range[0],
        y_range[1] - y_range[0],
        z_range[1] - z_range[0]
    )
    
    x_mid = (x_range[0] + x_range[1]) / 2
    y_mid = (y_range[0] + y_range[1]) / 2
    z_mid = (z_range[0] + z_range[1]) / 2
    
    return {
        'x': (x_mid - max_range/2, x_mid + max_range/2),
        'y': (y_mid - max_range/2, y_mid + max_range/2),
        'z': (z_mid - max_range/2, z_mid + max_range/2)
    }


def create_animation(joint_names, connections, frames, output_path=None, fps=30, rotate_camera=True):
    """Create and display/save the 3D skeleton animation."""
    
    # Calculate bounds
    bounds = calculate_data_bounds(frames)
    print(f"Data bounds - X: {bounds['x']}, Y: {bounds['y']}, Z: {bounds['z']}")
    
    # Setup figure
    fig = plt.figure(figsize=(10, 10), facecolor='#1a1a2e')
    ax = fig.add_subplot(111, projection='3d', facecolor='#1a1a2e')
    
    # Style settings
    bone_color = '#00d4ff'
    joint_color = '#ff6b6b'
    
    def update(num):
        ax.clear()
        
        # Set dark theme
        ax.set_facecolor('#1a1a2e')
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        ax.xaxis.pane.set_edgecolor('#333355')
        ax.yaxis.pane.set_edgecolor('#333355')
        ax.zaxis.pane.set_edgecolor('#333355')
        ax.grid(True, alpha=0.3, color='#444466')
        
        # Extract joint positions for current frame
        current_frame = frames[num]['joints']
        
        # Plot Bones (Connections)
        for conn in connections:
            j1_name, j2_name = joint_names[conn[0]], joint_names[conn[1]]
            
            if j1_name in current_frame and j2_name in current_frame:
                p1, p2 = current_frame[j1_name], current_frame[j2_name]
                ax.plot(
                    [p1['x'], p2['x']], 
                    [p1['y'], p2['y']], 
                    [p1['z'], p2['z']], 
                    color=bone_color, 
                    linewidth=3,
                    alpha=0.9
                )

        # Plot Joint Points
        xs = [p['x'] for p in current_frame.values()]
        ys = [p['y'] for p in current_frame.values()]
        zs = [p['z'] for p in current_frame.values()]
        ax.scatter(xs, ys, zs, color=joint_color, s=50, alpha=1.0, edgecolors='white', linewidths=0.5)

        # Set consistent axes
        ax.set_xlim(bounds['x'])
        ax.set_ylim(bounds['y'])
        ax.set_zlim(bounds['z'])
        
        # Labels with styling
        ax.set_xlabel('X', color='white', fontsize=10)
        ax.set_ylabel('Y', color='white', fontsize=10)
        ax.set_zlabel('Z', color='white', fontsize=10)
        ax.tick_params(colors='white', labelsize=8)
        
        ax.set_title(
            f"Climber 3D Reconstruction - Frame {num + 1}/{len(frames)}", 
            color='white', 
            fontsize=14, 
            fontweight='bold',
            pad=20
        )
        
        # Rotate camera slowly for dynamic view
        if rotate_camera:
            ax.view_init(elev=15, azim=num * 0.5)
        else:
            ax.view_init(elev=15, azim=45)

    # Calculate interval from fps
    interval = 1000 / fps  # milliseconds per frame
    
    print(f"Creating animation with {len(frames)} frames at {fps} FPS...")
    ani = FuncAnimation(fig, update, frames=len(frames), interval=interval, blit=False)
    
    if output_path:
        print(f"Saving animation to: {output_path}")
        print("This may take a while depending on the number of frames...")
        
        # Determine writer based on file extension
        if output_path.endswith('.gif'):
            ani.save(output_path, writer='pillow', fps=fps)
        else:
            ani.save(output_path, writer='ffmpeg', fps=fps, dpi=150,
                    metadata={'title': 'Climber 3D Reconstruction'})
        print(f"Animation saved successfully!")
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    app = SkeletonViewer()
    app.run()
