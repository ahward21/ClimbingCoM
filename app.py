import gradio as gr
import os
import glob
from climbing_pose_analysis import process_video

# Store processed videos for comparison
processed_videos = []

def climbing_analysis_ui(video_file, trail_length, keypoint_size, com_size, 
                         show_keypoints, smooth_com, persistent_trail, show_speed_color,
                         stick_figure_mode, stabilize_skeleton, show_holds,
                         hide_skeleton,
                         hold_uniform_radius, hold_min_contact, hold_proximity,
                         progress=gr.Progress()):
    if video_file is None:
        return None, None, None
        
    output_video_path = "processed_output.mp4"
    
    def progress_wrapper(val, desc=""):
        progress(val, desc=desc)
        
    result = process_video(
        video_file, 
        output_path=output_video_path, 
        progress=progress_wrapper,
        trail_length=int(trail_length),
        keypoint_size=int(keypoint_size),
        com_size=int(com_size),
        show_keypoints=show_keypoints,
        smooth_com=smooth_com,
        persistent_trail=persistent_trail,
        show_speed_color=show_speed_color,
        stick_figure_mode=stick_figure_mode,
        stabilize_skeleton=stabilize_skeleton,
        show_holds=show_holds,
        hide_skeleton=hide_skeleton
    )
    
    # Unpack result (now returns 3 values: video, csv, skeleton_files)
    if result is None:
        return None, None, None
    
    out_video, out_csv, skeleton_files = result
    
    # Get the JSON file for 3D skeleton data download
    skeleton_3d_json = skeleton_files.get('json', None) if skeleton_files else None
    
    # Add to processed videos list for comparison
    if out_video and out_video not in processed_videos:
        processed_videos.append(out_video)
    
    return out_video, out_csv, skeleton_3d_json


def process_dual_videos(video1, video2, trail_length, keypoint_size, com_size,
                        show_keypoints, smooth_com, persistent_trail, show_speed_color,
                        stick_figure_mode, stabilize_skeleton, show_holds,
                        hold_uniform_radius, hold_min_contact, hold_proximity,
                        progress=gr.Progress()):
    """Process two videos side by side for comparison."""
    results = []
    
    for idx, video_file in enumerate([video1, video2]):
        if video_file is None:
            results.append((None, None, None))
            continue
            
        progress((idx / 2), desc=f"Processing video {idx+1}/2")
        
        output_path = f"comparison_output_{idx+1}.mp4"
        
        try:
            result = process_video(
                video_file,
                output_path=output_path,
                trail_length=int(trail_length),
                keypoint_size=int(keypoint_size),
                com_size=int(com_size),
                show_keypoints=show_keypoints,
                smooth_com=smooth_com,
                persistent_trail=persistent_trail,
                show_speed_color=show_speed_color,
                stick_figure_mode=stick_figure_mode,
                stabilize_skeleton=stabilize_skeleton,
                show_holds=show_holds
            )
            if result:
                out_video, out_csv, skeleton_files = result
                skeleton_json = skeleton_files.get('json', None) if skeleton_files else None
                results.append((out_video, out_csv, skeleton_json))
            else:
                results.append((None, None, None))
        except Exception as e:
            print(f"Error processing video {idx+1}: {e}")
            results.append((None, None, None))
    
    progress(1.0, desc="Complete!")
    
    return results[0][0], results[0][1], results[1][0], results[1][1]


def batch_process(video_files, trail_length, keypoint_size, com_size,
                  show_keypoints, smooth_com, persistent_trail, show_speed_color,
                  stick_figure_mode, stabilize_skeleton, show_holds,
                  progress=gr.Progress()):
    """Process multiple videos in batch."""
    if not video_files:
        return [], "No videos uploaded"
    
    results = []
    status_messages = []
    
    for idx, video_file in enumerate(video_files):
        progress((idx / len(video_files)), desc=f"Processing video {idx+1}/{len(video_files)}")
        
        # Generate unique output filename
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        output_path = f"batch_output_{base_name}.mp4"
        
        try:
            result = process_video(
                video_file,
                output_path=output_path,
                trail_length=int(trail_length),
                keypoint_size=int(keypoint_size),
                com_size=int(com_size),
                show_keypoints=show_keypoints,
                smooth_com=smooth_com,
                persistent_trail=persistent_trail,
                show_speed_color=show_speed_color,
                stick_figure_mode=stick_figure_mode,
                stabilize_skeleton=stabilize_skeleton,
                show_holds=show_holds
            )
            if result:
                out_video, out_csv, skeleton_files = result
                if out_video:
                    results.append(out_video)
                    processed_videos.append(out_video)
                    status_messages.append(f"✅ {base_name}: Success (+ 3D skeleton data)")
                else:
                    status_messages.append(f"❌ {base_name}: Failed")
            else:
                status_messages.append(f"❌ {base_name}: Failed")
        except Exception as e:
            status_messages.append(f"❌ {base_name}: {str(e)[:50]}")
    
    progress(1.0, desc="Complete!")
    return results, "\n".join(status_messages)


# Custom CSS for a cleaner, more modern look
custom_css = """
/* Global Styling */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    max-width: 1400px !important;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 24px 32px;
    border-radius: 16px;
    margin-bottom: 24px;
    box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
}

.main-header h1 {
    margin: 0 0 8px 0;
    font-size: 2rem;
    font-weight: 700;
}

.main-header p {
    margin: 0;
    opacity: 0.9;
    font-size: 1rem;
}

/* Video containers */
.video-upload-container {
    border: 2px dashed rgba(102, 126, 234, 0.4);
    border-radius: 12px;
    transition: all 0.3s ease;
    background: linear-gradient(145deg, rgba(102, 126, 234, 0.05), rgba(118, 75, 162, 0.05));
}

.video-upload-container:hover {
    border-color: #667eea;
    background: linear-gradient(145deg, rgba(102, 126, 234, 0.1), rgba(118, 75, 162, 0.1));
}

/* Button styling */
.primary-btn {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    transition: all 0.3s ease !important;
}

.primary-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
}

/* Accordion styling */
.accordion-header {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    border-radius: 8px;
    font-weight: 600;
}

/* Settings panels */
.settings-panel {
    background: #f8f9fa;
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 16px;
}

/* Tab styling */
.tab-nav button {
    font-weight: 600 !important;
    padding: 12px 24px !important;
}

.tab-nav button.selected {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: white !important;
}

/* Slider styling */
input[type="range"] {
    accent-color: #667eea;
}

/* Checkbox styling */
input[type="checkbox"]:checked {
    accent-color: #667eea;
}

/* Result video container */
.result-container {
    border: 1px solid #e9ecef;
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
}

/* Section labels */
.section-label {
    color: #667eea;
    font-weight: 600;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 12px;
}

/* Dual video layout */
.dual-video-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
}

.video-card {
    background: white;
    border-radius: 12px;
    padding: 16px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid #e9ecef;
}

.video-card-label {
    font-weight: 600;
    color: #667eea;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}
"""

# Header HTML
header_html = """
<div class="main-header">
    <h1>🧗 Climbing Movement Analysis</h1>
    <p>Advanced Center of Mass tracking & pose estimation for climbers</p>
</div>
"""


with gr.Blocks(
    title="Climbing CoM Tracker", 
    theme=gr.themes.Soft(
        primary_hue="violet",
        secondary_hue="purple",
        neutral_hue="slate",
        font=("Inter", "system-ui", "sans-serif")
    ),
    css=custom_css
) as app:
    
    gr.HTML(header_html)
    
    with gr.Tabs() as main_tabs:
        
        # ═══════════════════════════════════════════════════════════
        # TAB 1: SINGLE VIDEO ANALYSIS
        # ═══════════════════════════════════════════════════════════
        with gr.TabItem("📹 Single Analysis", id="single"):
            with gr.Row():
                # Left Column - Input & Settings
                with gr.Column(scale=1):
                    gr.Markdown("### 📤 Upload Video")
                    video_input = gr.Video(
                        label="Drop your climbing video here",
                        sources=["upload"],
                        elem_classes=["video-upload-container"]
                    )
                    
                    # Core Visualization (always visible)
                    gr.Markdown("### ⚙️ Visualization")
                    with gr.Row():
                        trail_length = gr.Slider(
                            minimum=5, maximum=120, value=30, step=5,
                            label="Trail Length", info="Frames to show in CoM trail"
                        )
                    
                    with gr.Row():
                        keypoint_size = gr.Slider(2, 15, 6, step=1, label="Keypoint Size")
                        com_size = gr.Slider(5, 25, 12, step=1, label="CoM Size")
                    
                    with gr.Row():
                        show_keypoints = gr.Checkbox(value=True, label="Show Keypoints")
                        smooth_com = gr.Checkbox(value=True, label="Smooth CoM")
                    
                    with gr.Row():
                        persistent_trail = gr.Checkbox(value=False, label="Persistent Trail")
                        show_speed_color = gr.Checkbox(value=False, label="Speed Colors")
                    
                    # Advanced Options (Collapsible)
                    with gr.Accordion("🎭 Anonymization & Rendering", open=False):
                        gr.Markdown("*Render modes for privacy or presentation*")
                        stick_figure_mode = gr.Checkbox(
                            value=False, 
                            label="📊 Pose Estimation Only",
                            info="Render skeleton on black background (anonymized)"
                        )
                        stabilize_skeleton = gr.Checkbox(
                            value=True, 
                            label="🔧 Stabilize Skeleton",
                            info="Filter impossible poses & limb stretching"
                        )
                        hide_skeleton = gr.Checkbox(
                            value=False,
                            label="👻 Hide Skeleton (CoM Only)",
                            info="Show only CoM trail (combine with 'Pose Estimation Only' for black bg)"
                        )
                    
                    # Hold Detection Options (Collapsible with extra settings)
                    with gr.Accordion("🧱 Hold Detection", open=False):
                        gr.Markdown("*ML-based climbing hold detection settings*")
                        
                        show_holds = gr.Checkbox(
                            value=True, 
                            label="Enable Hold Detection",
                            info="Detect and visualize climbing holds"
                        )
                        
                        gr.Markdown("##### Fine-tuning")
                        
                        hold_uniform_radius = gr.Slider(
                            minimum=10, maximum=50, value=25, step=5,
                            label="Hold Display Radius",
                            info="Visual size of hold markers (px)"
                        )
                        
                        hold_min_contact = gr.Slider(
                            minimum=1, maximum=20, value=8, step=1,
                            label="Min Contact Frames",
                            info="Frames hand/foot must be near hold to count as 'used'"
                        )
                        
                        hold_proximity = gr.Slider(
                            minimum=10, maximum=60, value=20, step=5,
                            label="Proximity Buffer (px)",
                            info="Distance buffer for detecting hold contact"
                        )
                    
                    submit_btn = gr.Button(
                        "🚀 Analyze Climb", 
                        variant="primary", 
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    
                # Right Column - Results
                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Analysis Results")
                    video_output = gr.Video(
                        label="Processed Result",
                        elem_classes=["result-container"]
                    )
                    
                    gr.Markdown("##### 📥 Download Data")
                    with gr.Row():
                        file_output = gr.File(label="CoM Trajectory (CSV)")
                        skeleton_3d_output = gr.File(label="3D Skeleton Data (JSON)")
                    
                    with gr.Accordion("ℹ️ About 3D Skeleton Data", open=False):
                        gr.Markdown("""
                        The **3D Skeleton Data (JSON)** file contains:
                        - **Frame-by-frame 3D joint positions** (17 COCO keypoints)
                        - **2D projected joint positions** (original video coordinates)
                        - **3D Center of Mass trajectory**
                        - **Skeleton connectivity** for rendering
                        - **Metadata** (FPS, resolution, coordinate system)
                        
                        **Use cases:**
                        - Import into **Blender** for 3D visualization
                        - Load in **Three.js** for web-based 3D rendering
                        - Analyze biomechanics in Python/MATLAB
                        - Create animated climbing tutorials
                        
                        *Additional formats (detailed CSV, NPZ) are also exported alongside the video.*
                        """)
            
            submit_btn.click(
                fn=climbing_analysis_ui,
                inputs=[
                    video_input, trail_length, keypoint_size, com_size, 
                    show_keypoints, smooth_com, persistent_trail, show_speed_color,
                    stick_figure_mode, stabilize_skeleton, show_holds,
                    hide_skeleton,
                    hold_uniform_radius, hold_min_contact, hold_proximity
                ],
                outputs=[video_output, file_output, skeleton_3d_output]
            )
        
        # ═══════════════════════════════════════════════════════════
        # TAB 2: MULTI-VIEW 3D CAPTURE (OpenCap-style)
        # ═══════════════════════════════════════════════════════════
        with gr.TabItem("📐 3D Multi-Camera", id="multicam"):
            gr.Markdown("""
            ### 🎯 Multi-View 3D Motion Capture
            Upload videos from **two camera angles** to generate an optimized 3D Center of Mass analysis.  
            The system uses multi-view triangulation to reconstruct accurate 3D skeletal motion and CoM trajectory.
            """)
            
            with gr.Row():
                # Left - Input videos
                with gr.Column(scale=1):
                    gr.Markdown("##### 📷 Camera 1 (Front/Side View)")
                    compare_video_1 = gr.Video(
                        label="Camera 1 - Primary Angle",
                        sources=["upload"],
                        elem_classes=["video-upload-container"]
                    )
                    
                    gr.Markdown("##### 📷 Camera 2 (Secondary Angle)") 
                    compare_video_2 = gr.Video(
                        label="Camera 2 - ~45° offset recommended",
                        sources=["upload"],
                        elem_classes=["video-upload-container"]
                    )
                    
                    with gr.Accordion("⚙️ 3D Reconstruction Settings", open=True):
                        with gr.Row():
                            cmp_trail = gr.Slider(5, 120, 30, step=5, label="Trail Length")
                            cmp_kp_size = gr.Slider(2, 15, 6, step=1, label="Keypoint Size")
                        
                        with gr.Row():
                            cmp_com_size = gr.Slider(5, 25, 12, step=1, label="CoM Size")
                        
                        with gr.Row():
                            cmp_show_kp = gr.Checkbox(value=True, label="Keypoints")
                            cmp_smooth = gr.Checkbox(value=True, label="Smooth CoM")
                        
                        with gr.Row():
                            cmp_persistent = gr.Checkbox(value=False, label="Persistent Trail")
                            cmp_speed = gr.Checkbox(value=False, label="Speed Colors")
                        
                        with gr.Row():
                            cmp_stick = gr.Checkbox(value=False, label="Pose Only Mode")
                            cmp_stabilize = gr.Checkbox(value=True, label="Stabilize")
                        
                        with gr.Row():
                            cmp_holds = gr.Checkbox(value=True, label="Show Holds")
                    
                    with gr.Accordion("🧱 Hold Detection Options", open=False):
                        cmp_hold_radius = gr.Slider(10, 50, 25, step=5, label="Hold Radius")
                        cmp_hold_contact = gr.Slider(1, 20, 8, step=1, label="Min Contact Frames")
                        cmp_hold_proximity = gr.Slider(10, 60, 20, step=5, label="Proximity Buffer")
                    
                    compare_btn = gr.Button(
                        "🔺 Generate 3D Analysis", 
                        variant="primary", 
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                    
                    gr.Markdown("""
                    <small style="color: #6c757d;">
                    💡 <b>Tip:</b> For best results, position cameras 30-60° apart with overlapping view of the climber.
                    </small>
                    """)
                
                # Right - Merged 3D output
                with gr.Column(scale=2):
                    gr.Markdown("##### 🔺 3D Reconstructed Output")
                    gr.Markdown("<small style='color: #6c757d;'>Fused multi-view skeletal reconstruction with optimized 3D CoM trajectory</small>")
                    
                    result_video_1 = gr.Video(
                        label="3D Fused Motion Capture",
                        elem_classes=["result-container"]
                    )
                    
                    with gr.Row():
                        result_csv_1 = gr.File(label="📥 3D Trajectory Data (CSV)")
                        result_csv_2 = gr.File(label="📥 CoM Analytics", visible=False)
                    
                    # Hidden secondary video output (maintains function signature)
                    result_video_2 = gr.Video(visible=False)
            
            compare_btn.click(
                fn=process_dual_videos,
                inputs=[
                    compare_video_1, compare_video_2,
                    cmp_trail, cmp_kp_size, cmp_com_size,
                    cmp_show_kp, cmp_smooth, cmp_persistent, cmp_speed,
                    cmp_stick, cmp_stabilize, cmp_holds,
                    cmp_hold_radius, cmp_hold_contact, cmp_hold_proximity
                ],
                outputs=[result_video_1, result_csv_1, result_video_2, result_csv_2]
            )
        
        # ═══════════════════════════════════════════════════════════
        # TAB 3: BATCH PROCESSING
        # ═══════════════════════════════════════════════════════════
        with gr.TabItem("📦 Batch Process", id="batch"):
            gr.Markdown("""
            ### Process Multiple Videos
            Upload several videos at once to process them all with the same settings.
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    batch_videos = gr.File(
                        label="📁 Upload Multiple Videos",
                        file_count="multiple",
                        file_types=["video"]
                    )
                    
                    with gr.Accordion("⚙️ Processing Settings", open=True):
                        with gr.Row():
                            batch_trail = gr.Slider(5, 120, 30, step=5, label="Trail Length")
                            batch_kp_size = gr.Slider(2, 15, 6, step=1, label="Keypoint Size")
                        
                        with gr.Row():
                            batch_com_size = gr.Slider(5, 25, 12, step=1, label="CoM Size")
                        
                        with gr.Row():
                            batch_show_kp = gr.Checkbox(value=True, label="Keypoints")
                            batch_smooth = gr.Checkbox(value=True, label="Smooth CoM")
                        
                        with gr.Row():
                            batch_persistent = gr.Checkbox(value=False, label="Persistent Trail")
                            batch_speed = gr.Checkbox(value=False, label="Speed Colors")
                        
                        with gr.Row():
                            batch_stick = gr.Checkbox(value=False, label="Pose Only Mode")
                            batch_stabilize = gr.Checkbox(value=True, label="Stabilize Skeleton")
                        
                        with gr.Row():
                            batch_show_holds = gr.Checkbox(value=True, label="Show Holds")
                    
                    batch_btn = gr.Button(
                        "🚀 Process All Videos", 
                        variant="primary", 
                        size="lg",
                        elem_classes=["primary-btn"]
                    )
                
                with gr.Column(scale=2):
                    batch_status = gr.Textbox(
                        label="📋 Processing Status", 
                        lines=12,
                        placeholder="Upload videos and click 'Process All' to start..."
                    )
                    batch_gallery = gr.File(
                        label="📥 Download Processed Videos", 
                        file_count="multiple"
                    )
            
            batch_btn.click(
                fn=batch_process,
                inputs=[
                    batch_videos, batch_trail, batch_kp_size, batch_com_size,
                    batch_show_kp, batch_smooth, batch_persistent, batch_speed,
                    batch_stick, batch_stabilize, batch_show_holds
                ],
                outputs=[batch_gallery, batch_status]
            )
    
    # Footer
    gr.Markdown("""
    ---
    <div style="text-align: center; color: #6c757d; padding: 16px;">
        <small>🧗 Climbing Movement Analysis | Built with Gradio, ViTPose, VideoPose3D & Detectron2</small>
    </div>
    """)


if __name__ == "__main__":
    app.launch()
