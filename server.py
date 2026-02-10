"""
ClimbAnalytics - Self-hosted Web Server
Flask-based web server for climbing center of mass analysis.
"""

import os
import sys
import uuid
import json
import logging
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, send_from_directory, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import threading

# Import the analysis module
from climbing_pose_analysis import process_video

# Setup logging - output to both console and file
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('server.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# Ensure stdout is not buffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

app = Flask(__name__, static_folder='static')
CORS(app)

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

# Active jobs tracking
active_jobs = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_opensim_trc(json_path, output_folder, session_name):
    """
    Generate OpenSim .trc file from 3D skeleton JSON data.
    
    TRC (Track Row Column) format is the standard for OpenSim marker data.
    This maps our COCO skeleton joints to OpenSim-compatible marker names.
    """
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        frames = data.get('frames', [])
        if not frames:
            return None
        
        # Get metadata
        fps = data.get('metadata', {}).get('fps', 30)
        
        # COCO keypoint names mapped to OpenSim-friendly marker names
        marker_names = [
            'Nose', 'LEye', 'REye', 'LEar', 'REar',
            'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
            'LWrist', 'RWrist', 'LHip', 'RHip',
            'LKnee', 'RKnee', 'LAnkle', 'RAnkle'
        ]
        
        # Build TRC content
        trc_lines = []
        
        # Header line 1
        trc_lines.append(f"PathFileType\t4\t(X/Y/Z)\t{session_name}.trc")
        
        # Header line 2
        trc_lines.append(f"DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames")
        
        # Header line 3
        num_frames = len(frames)
        num_markers = len(marker_names)
        trc_lines.append(f"{fps}\t{fps}\t{num_frames}\t{num_markers}\tm\t{fps}\t1\t{num_frames}")
        
        # Header line 4 - marker names
        marker_header = "Frame#\tTime\t"
        for name in marker_names:
            marker_header += f"{name}\t\t\t"  # X, Y, Z columns
        trc_lines.append(marker_header.rstrip('\t'))
        
        # Header line 5 - coordinate labels
        coord_header = "\t\t"
        for i, name in enumerate(marker_names):
            coord_header += f"X{i+1}\tY{i+1}\tZ{i+1}\t"
        trc_lines.append(coord_header.rstrip('\t'))
        
        # Empty line
        trc_lines.append("")
        
        # Data rows
        for frame in frames:
            frame_num = frame.get('frame', 0)
            time_sec = frame_num / fps
            
            row = f"{frame_num}\t{time_sec:.6f}\t"
            
            joints_3d = frame.get('joints_3d', [])
            for i in range(len(marker_names)):
                if i < len(joints_3d):
                    x, y, z = joints_3d[i]
                    # Convert from mm to meters if needed, and adjust coordinate system
                    # OpenSim uses: X=forward, Y=up, Z=right
                    row += f"{x:.6f}\t{y:.6f}\t{z:.6f}\t"
                else:
                    row += "NaN\tNaN\tNaN\t"
            
            trc_lines.append(row.rstrip('\t'))
        
        # Write TRC file
        trc_path = output_folder / f"{session_name}_opensim.trc"
        with open(trc_path, 'w') as f:
            f.write('\n'.join(trc_lines))
        
        logger.info(f"Generated OpenSim TRC: {trc_path}")
        return str(trc_path)
        
    except Exception as e:
        logger.error(f"Failed to generate OpenSim TRC: {e}")
        return None

@app.route('/')
def index():
    """Serve the main application page."""
    return send_from_directory('static', 'index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from the static folder."""
    return send_from_directory('static', filename)

@app.route('/api/upload', methods=['POST'])
def upload_video():
    """Handle video upload."""
    print(f"[UPLOAD] Received upload request", flush=True)
    logger.info("Received upload request")
    
    if 'video1' not in request.files:
        logger.error("No video file in request")
        return jsonify({'error': 'No video file provided'}), 400
    
    file1 = request.files['video1']
    file2 = request.files.get('video2')  # Optional second video
    
    if file1.filename == '':
        logger.error("Empty filename")
        return jsonify({'error': 'No selected file'}), 400
    
    if file1 and allowed_file(file1.filename):
        # Generate unique job ID
        job_id = str(uuid.uuid4())
        
        # Save uploaded files
        filename1 = secure_filename(f"{job_id}_angle1_{file1.filename}")
        filepath1 = UPLOAD_FOLDER / filename1
        file1.save(str(filepath1))
        
        filepath2 = None
        if file2 and file2.filename and allowed_file(file2.filename):
            filename2 = secure_filename(f"{job_id}_angle2_{file2.filename}")
            filepath2 = UPLOAD_FOLDER / filename2
            file2.save(str(filepath2))
        
        # Store job info
        active_jobs[job_id] = {
            'status': 'uploaded',
            'progress': 0,
            'message': 'Files uploaded successfully',
            'video1': str(filepath1),
            'video2': str(filepath2) if filepath2 else None,
            'result_video': None,
            'result_csv': None,
            'result_csv': None,
            'result_conf_csv': None,
            'result_skeleton': None,
            'metrics': None,
            'settings': {},
            'logs': []
        }
        
        print(f"[UPLOAD] Job {job_id}: Video uploaded - {filename1}", flush=True)
        logger.info(f"Job {job_id}: Video uploaded - {filename1}")
        
        return jsonify({
            'job_id': job_id,
            'message': 'Files uploaded successfully',
            'has_second_video': filepath2 is not None
        })
    
    logger.error(f"Invalid file type: {file1.filename}")
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/api/analyze/<job_id>', methods=['POST'])
def start_analysis(job_id):
    """Start the analysis for a given job."""
    print(f"[ANALYZE] Starting analysis for job {job_id}", flush=True)
    
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    if job['status'] == 'processing':
        return jsonify({'error': 'Analysis already in progress'}), 400
    
    # Get settings from request
    settings = request.json or {}
    job['settings'] = settings
    
    print(f"[ANALYZE] Job {job_id}: Settings: {settings}", flush=True)
    logger.info(f"Job {job_id}: Starting analysis with settings: {settings}")
    
    # Start processing in background thread
    thread = threading.Thread(target=run_analysis, args=(job_id,))
    thread.daemon = True
    thread.start()
    
    job['status'] = 'processing'
    job['progress'] = 0
    job['message'] = 'Starting analysis...'
    
    return jsonify({'message': 'Analysis started', 'job_id': job_id})

def run_analysis(job_id):
    """Run the video analysis in a background thread."""
    job = active_jobs[job_id]
    settings = job.get('settings', {})
    
    def add_log(msg):
        """Add message to job's log."""
        timestamp = time.strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {msg}"
        job['logs'].append(log_entry)
        print(f"[JOB {job_id[:8]}] {msg}", flush=True)
        logger.info(f"[{job_id}] {msg}")
    
    try:
        add_log("Analysis started")
        
        # Collect video inputs
        videos_to_process = []
        if job.get('video1'):
            videos_to_process.append({'path': job['video1'], 'label': 'angle1'})
        if job.get('video2'):
            videos_to_process.append({'path': job['video2'], 'label': 'angle2'})
            
        if not videos_to_process:
            raise ValueError("No video files found in job")

        # Progress callback wrapper
        # We split progress: 0-90% for processing (divided by num videos), 90-100% for finalization
        total_videos = len(videos_to_process)
        
        def make_progress_callback(video_idx, total_vids):
            base_progress = (video_idx / total_vids) * 90
            chunk_size = 90 / total_vids
            
            def callback(value, desc=""):
                current_percent = base_progress + (value * chunk_size)
                job['progress'] = int(current_percent)
                job['message'] = f"[{video_idx+1}/{total_vids}] {desc}"
                add_log(f"Progress: {job['progress']}% - {desc}")
            return callback
        
        # Create organized output folder
        # Use job ID as base or first video name
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        session_name = settings.get('session_name', f"session_{job_id[:8]}")
        folder_name = f"{timestamp}_{session_name}"
        session_folder = OUTPUT_FOLDER / folder_name
        session_folder.mkdir(exist_ok=True)
        
        job['session_folder'] = str(session_folder)
        job['results'] = {} # Store results per label
        
        # Get visualization settings
        trail_length = settings.get('trail_length', 30)
        keypoint_size = settings.get('keypoint_size', 6)
        com_size = settings.get('com_size', 12)
        show_keypoints = settings.get('show_keypoints', True)
        smooth_com = settings.get('smooth_com', True)
        persistent_trail = settings.get('persistent_trail', False)
        show_speed_color = settings.get('show_speed_color', False)
        stick_figure_mode = settings.get('stick_figure_mode', False)
        stabilize_skeleton = settings.get('stabilize_skeleton', True)
        
        add_log(f"Session: {folder_name}")
        add_log(f"Processing {total_videos} video(s)")
        
        # --- PROCESS LOOP ---
        for idx, vid_info in enumerate(videos_to_process):
            video_path = vid_info['path']
            label = vid_info['label']
            
            add_log(f"--- Processing Video {idx+1}/{total_videos} ({label}) ---")
            
            # Define unique output name
            # e.g. session_name_angle1_analyzed.mp4
            base_name = f"{session_name}_{label}"
            output_video_path = session_folder / f"{base_name}_analyzed.mp4"
            
            cb = make_progress_callback(idx, total_videos)
            
            result = process_video(
                video_path=video_path,
                output_path=str(output_video_path),
                progress=cb,
                trail_length=int(trail_length),
                keypoint_size=int(keypoint_size),
                com_size=int(com_size),
                show_keypoints=show_keypoints,
                smooth_com=smooth_com,
                persistent_trail=persistent_trail,
                show_speed_color=show_speed_color,
                stick_figure_mode=stick_figure_mode,
                stabilize_skeleton=stabilize_skeleton,
                hide_skeleton=settings.get('hide_skeleton', False),
                show_holds=False
            )
            
            if result:
                out_video, out_csv, skeleton_files, metrics = result
                
                # Store results for this angle
                angle_results = {
                    'video': str(output_video_path),
                    'metrics': metrics,
                    'csv': None,
                    'skeleton': None,
                    'conf_csv': None,
                    'trc': None
                }
                
                add_log(f"Accuracy ({label}): {metrics.get('model_accuracy_percent', 0):.1f}%")
                
                # File Management & Renaming
                import shutil
                
                # 1. Rename Basic CSV
                if out_csv and os.path.exists(out_csv):
                    new_csv = session_folder / f"{base_name}_com_trajectory.csv"
                    shutil.move(out_csv, new_csv)
                    angle_results['csv'] = str(new_csv)
                
                # 2. Rename JSON Skeleton
                if skeleton_files and skeleton_files.get('json') and os.path.exists(skeleton_files['json']):
                    new_json = session_folder / f"{base_name}_skeleton_3d.json"
                    shutil.move(skeleton_files['json'], new_json)
                    angle_results['skeleton'] = str(new_json)
                    
                    # Generate TRC
                    add_log(f"Generating OpenSim TRC for {label}...")
                    trc_path = generate_opensim_trc(str(new_json), session_folder, base_name)
                    if trc_path:
                        angle_results['trc'] = trc_path
                
                # 3. Rename other skeleton files
                if skeleton_files:
                    for key in ['npz', 'detailed_csv', 'csv_confidence']:
                        if skeleton_files.get(key) and os.path.exists(skeleton_files[key]):
                            ext = 'npz' if key == 'npz' else 'csv'
                            suffix = ''
                            if key == 'npz': suffix = '_skeleton'
                            elif key == 'detailed_csv': suffix = '_skeleton_detailed'
                            elif key == 'csv_confidence': suffix = '_confidence_levels'
                            
                            new_path = session_folder / f"{base_name}{suffix}.{ext}"
                            shutil.move(skeleton_files[key], new_path)
                            
                            if key == 'csv_confidence':
                                angle_results['conf_csv'] = str(new_path)
                
                job['results'][label] = angle_results
            else:
                add_log(f"Failed to process {label}")
                job['results'][label] = {'error': 'Processing failed'}
        
        # --- FINALIZATION ---
        job['progress'] = 100
        job['status'] = 'complete'
        job['message'] = 'Analysis complete!'
        
        # Set primary results for backward compatibility (previewing angle 1)
        if 'angle1' in job['results'] and 'error' not in job['results']['angle1']:
            res = job['results']['angle1']
            job['result_video'] = res.get('video')
            job['result_csv'] = res.get('csv')
            job['result_skeleton'] = res.get('skeleton')
            job['result_conf_csv'] = res.get('conf_csv')
            job['result_trc'] = res.get('trc')
            job['metrics'] = res.get('metrics')
        
        add_log(f"SUCCESS! All files saved to: {session_folder}")
            
    except Exception as e:
        error_msg = str(e)
        logger.error(f"[{job_id}] Analysis error: {e}", exc_info=True)
        job['status'] = 'error'
        job['message'] = f'Analysis failed: {error_msg}'
        add_log(f"ERROR: {error_msg}")

def get_file_path(job_id, file_key, angle=None):
    """Helper to get file path for a specific angle."""
    job = active_jobs[job_id]
    
    # If using new results structure
    if 'results' in job:
        # If angle specified, try to get it
        if angle and angle in job['results']:
            res = job['results'][angle]
            if 'error' not in res and res.get(file_key):
                return res[file_key]
        
        # Default to angle1 if it exists
        if 'angle1' in job['results'] and 'error' not in job['results']['angle1']:
             res = job['results']['angle1']
             if res.get(file_key):
                 return res[file_key]
                 
        # Fallback to any valid result if angle1 not found
        for lbl, res in job['results'].items():
            if 'error' not in res and res.get(file_key):
                return res[file_key]
                
    # Fallback to legacy/root keys
    root_key_map = {
        'video': 'result_video',
        'csv': 'result_csv',
        'skeleton': 'result_skeleton',
        'trc': 'result_trc',
        'conf_csv': 'result_conf_csv'
    }
    
    if file_key in root_key_map:
        return job.get(root_key_map[file_key])
        
    return None

@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get the status of an analysis job."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    return jsonify({
        'status': job['status'],
        'progress': job['progress'],
        'message': job['message'],
        'has_result': job.get('result_video') is not None,
        'results': job.get('results'),
        'metrics': job.get('metrics'),
        'logs': job.get('logs', [])[-10:]  # Last 10 log entries
    })

@app.route('/api/logs/<job_id>')
def get_logs(job_id):
    """Get all logs for a job."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = active_jobs[job_id]
    return jsonify({'logs': job.get('logs', [])})

@app.route('/api/result/<job_id>/video')
def get_result_video(job_id):
    """Get the result video."""
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    angle = request.args.get('angle')
    video_path = get_file_path(job_id, 'video', angle)
        
    if not video_path or not os.path.exists(video_path):
        return jsonify({'error': 'No video result available'}), 404
    
    print(f"[DOWNLOAD] Serving video: {video_path}", flush=True)
    return send_file(video_path, mimetype='video/mp4')

@app.route('/api/result/<job_id>/csv')
def get_result_csv(job_id):
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    angle = request.args.get('angle')
    path = get_file_path(job_id, 'csv', angle)
    
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
        
    filename = f'climbing_trajectory_{job_id[:8]}_{angle or "main"}.csv'
    return send_file(path, mimetype='text/csv', as_attachment=True, download_name=filename)

@app.route('/api/result/<job_id>/skeleton')
def get_result_skeleton(job_id):
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    angle = request.args.get('angle')
    path = get_file_path(job_id, 'skeleton', angle)
    
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
        
    filename = f'skeleton_3d_{job_id[:8]}_{angle or "main"}.json'
    return send_file(path, mimetype='application/json', as_attachment=True, download_name=filename)

@app.route('/api/result/<job_id>/trc')
def get_result_trc(job_id):
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    angle = request.args.get('angle')
    path = get_file_path(job_id, 'trc', angle)
    
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
        
    filename = f'opensim_{job_id[:8]}_{angle or "main"}.trc'
    return send_file(path, mimetype='text/plain', as_attachment=True, download_name=filename)

@app.route('/api/result/<job_id>/confidence')
def get_result_confidence(job_id):
    if job_id not in active_jobs:
        return jsonify({'error': 'Job not found'}), 404
        
    angle = request.args.get('angle')
    path = get_file_path(job_id, 'conf_csv', angle)
    
    if not path or not os.path.exists(path):
        return jsonify({'error': 'File not found'}), 404
        
    filename = f'confidence_levels_{job_id[:8]}_{angle or "main"}.csv'
    return send_file(path, mimetype='text/csv', as_attachment=True, download_name=filename)

@app.route('/api/jobs')
def list_jobs():
    """List all jobs and their statuses."""
    jobs_list = []
    for job_id, job in active_jobs.items():
        jobs_list.append({
            'job_id': job_id,
            'status': job['status'],
            'progress': job['progress'],
            'message': job['message']
        })
    return jsonify(jobs_list)

@app.route('/api/health')
def health_check():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'active_jobs': len(active_jobs)})

@app.route('/api/open-folder', methods=['POST'])
def open_output_folder():
    """Open the outputs folder in Windows Explorer."""
    import subprocess
    folder_path = OUTPUT_FOLDER.absolute()
    try:
        # Open folder in Windows Explorer
        subprocess.Popen(f'explorer "{folder_path}"')
        return jsonify({'success': True, 'path': str(folder_path)})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/output-path')
def get_output_path():
    """Get the absolute path to the outputs folder."""
    return jsonify({'path': str(OUTPUT_FOLDER.absolute())})

if __name__ == '__main__':
    print()
    print("=" * 60)
    print("  ClimbAnalytics - Self-Hosted Web Server")
    print("  Center of Mass Trajectory Analysis")
    print("=" * 60)
    print()
    print("  Features:")
    print("    ✓ Single video analysis")
    print("    ✓ Multi-camera 3D reconstruction")
    print("    ✓ Batch processing")
    print("    ✓ Pose-only (anonymized) mode")
    print("    ✓ CoM trajectory visualization")
    print("    ✓ Speed-based coloring")
    print("    ✓ JSON/CSV/Video export")
    print()
    print("  Server running at: http://localhost:5000")
    print("  Press Ctrl+C to stop the server.")
    print()
    print("  Console will show processing progress...")
    print("=" * 60)
    print()
    
    # Flush to ensure immediate output
    sys.stdout.flush()
    
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
