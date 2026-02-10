"""
3D Skeleton Viewer Module
Integrates Plotly 3D visualization into ClimbAnalytics
"""

import json
import numpy as np

# ==================== CONFIGURATION ====================
BODY_COLORS = {
    'head': '#FF6B9D', 'torso': '#00D4FF',
    'left_arm': '#FFD93D', 'right_arm': '#6BCB77',
    'left_leg': '#9D65C9', 'right_leg': '#FF8C42',
    'com': '#FF3366',
}

BODY_JOINTS = [
    'nose', 'left_shoulder', 'right_shoulder',
    'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist',
    'left_hip', 'right_hip', 'left_knee', 'right_knee',
    'left_ankle', 'right_ankle',
]

SKELETON_CONNECTIONS = [
    ('nose', 'left_shoulder'), ('nose', 'right_shoulder'),
    ('left_shoulder', 'right_shoulder'),
    ('left_shoulder', 'left_elbow'), ('left_elbow', 'left_wrist'),
    ('right_shoulder', 'right_elbow'), ('right_elbow', 'right_wrist'),
    ('left_shoulder', 'left_hip'), ('right_shoulder', 'right_hip'),
    ('left_hip', 'right_hip'),
    ('left_hip', 'left_knee'), ('left_knee', 'left_ankle'),
    ('right_hip', 'right_knee'), ('right_knee', 'right_ankle'),
]

JOINT_BODY_PARTS = {
    'nose': 'head',
    'left_shoulder': 'torso', 'right_shoulder': 'torso',
    'left_elbow': 'left_arm', 'right_elbow': 'right_arm',
    'left_wrist': 'left_arm', 'right_wrist': 'right_arm',
    'left_hip': 'torso', 'right_hip': 'torso',
    'left_knee': 'left_leg', 'right_knee': 'right_leg',
    'left_ankle': 'left_leg', 'right_ankle': 'right_leg',
}

DEPTH_OFFSETS = {
    'nose': 0.15, 'left_shoulder': 0.05, 'right_shoulder': -0.05,
    'left_elbow': 0.1, 'right_elbow': -0.1,
    'left_wrist': 0.15, 'right_wrist': -0.15,
    'left_hip': 0.03, 'right_hip': -0.03,
    'left_knee': 0.08, 'right_knee': -0.08,
    'left_ankle': 0.1, 'right_ankle': -0.1,
}


def get_bone_color(j1, j2):
    p1, p2 = JOINT_BODY_PARTS.get(j1, 'torso'), JOINT_BODY_PARTS.get(j2, 'torso')
    if p1 == p2: return BODY_COLORS[p1]
    return BODY_COLORS['torso'] if 'torso' in [p1, p2] else BODY_COLORS[p1]


def extract_positions(frames):
    n = len(frames)
    pos = np.zeros((n, len(BODY_JOINTS), 3))
    com = np.zeros((n, 3))
    depths = []
    
    for i, f in enumerate(frames):
        j2d = f.get('joints_2d', {})
        for j, name in enumerate(BODY_JOINTS):
            if name in j2d:
                pos[i, j, 0] = j2d[name]['x']
                pos[i, j, 2] = -j2d[name]['y']
                pos[i, j, 1] = DEPTH_OFFSETS.get(name, 0) * 100
        
        if 'left_hip' in j2d and 'right_hip' in j2d:
            com[i, 0] = (j2d['left_hip']['x'] + j2d['right_hip']['x']) / 2
            com[i, 2] = -(j2d['left_hip']['y'] + j2d['right_hip']['y']) / 2
        
        depths.append(f.get('center_of_mass', {}).get('y', 0))
    
    if depths:
        d = np.array(depths)
        std = np.std(d)
        if std > 0.001:
            com[:, 1] = (d - np.mean(d)) / std * 0.15
    
    return pos, com


def normalize(pos, com):
    flat = pos.reshape(-1, 3)
    valid = flat[np.any(flat != 0, axis=1)]
    if len(valid) == 0:
        return pos, com
    
    ctr = (np.min(valid, axis=0) + np.max(valid, axis=0)) / 2
    scl = max(np.max(valid[:, 0]) - np.min(valid[:, 0]), np.max(valid[:, 2]) - np.min(valid[:, 2]))
    if scl < 0.001: scl = 1.0
    
    pos = (pos - ctr) / scl * 2
    com[:, 0] = (com[:, 0] - ctr[0]) / scl * 2
    com[:, 2] = (com[:, 2] - ctr[2]) / scl * 2
    
    # Rotate 30° for overhang
    a = np.radians(30)
    c, s = np.cos(a), np.sin(a)
    for i in range(len(pos)):
        for j in range(pos.shape[1]):
            y, z = pos[i, j, 1], pos[i, j, 2]
            pos[i, j, 1], pos[i, j, 2] = y*c - z*s, y*s + z*c
        y, z = com[i, 1], com[i, 2]
        com[i, 1], com[i, 2] = y*c - z*s, y*s + z*c
    
    return pos, com


def generate_plotly_json(json_path, step=5):
    """
    Generate Plotly figure data from skeleton JSON.
    Returns a JSON string that can be rendered client-side.
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        meta = data.get('metadata', {})
        frames_data = data.get('frames', [])
        
        if not frames_data:
            return None
        
        pos, com = normalize(*extract_positions(frames_data))
        jidx = {n: i for i, n in enumerate(BODY_JOINTS)}
        n_frames = len(frames_data)
        indices = list(range(0, n_frames, step))
        
        fig = go.Figure()
        fi = 0
        
        # Bones
        for j1, j2 in SKELETON_CONNECTIONS:
            i1, i2 = jidx[j1], jidx[j2]
            fig.add_trace(go.Scatter3d(
                x=[pos[fi, i1, 0], pos[fi, i2, 0]],
                y=[pos[fi, i1, 1], pos[fi, i2, 1]],
                z=[pos[fi, i1, 2], pos[fi, i2, 2]],
                mode='lines', line=dict(color=get_bone_color(j1, j2), width=12),
                hoverinfo='skip', showlegend=False
            ))
        
        # Joints
        for j, name in enumerate(BODY_JOINTS):
            fig.add_trace(go.Scatter3d(
                x=[pos[fi, j, 0]], y=[pos[fi, j, 1]], z=[pos[fi, j, 2]],
                mode='markers', marker=dict(size=12, color=BODY_COLORS[JOINT_BODY_PARTS.get(name, 'torso')],
                                            line=dict(width=2, color='white')),
                hovertemplate=f'<b>{name}</b><extra></extra>', showlegend=False
            ))
        
        # COM
        fig.add_trace(go.Scatter3d(
            x=[com[fi, 0]], y=[com[fi, 1]], z=[com[fi, 2]],
            mode='markers', marker=dict(size=16, color=BODY_COLORS['com'], symbol='diamond'),
            name='Center of Mass', showlegend=True
        ))
        
        # Trail
        fig.add_trace(go.Scatter3d(
            x=com[:fi+1, 0].tolist(), y=com[:fi+1, 1].tolist(), z=com[:fi+1, 2].tolist(),
            mode='lines', line=dict(color=BODY_COLORS['com'], width=4),
            opacity=0.6, name='COM Trail', showlegend=True
        ))
        
        # Wall
        wx = np.linspace(-1.8, 1.8, 10)
        wh = np.linspace(-1.8, 1.8, 10)
        WX, WH = np.meshgrid(wx, wh)
        a = np.radians(-30)
        fig.add_trace(go.Surface(
            x=WX.tolist(), y=(0.6 + WH * np.sin(a)).tolist(), z=(WH * np.cos(a)).tolist(),
            colorscale=[[0, '#2a3a4a'], [1, '#3a4a5a']],
            showscale=False, opacity=0.4, hoverinfo='skip'
        ))
        
        # Animation frames
        anim_frames = []
        for fidx, fi in enumerate(indices):
            frame_data = []
            for j1, j2 in SKELETON_CONNECTIONS:
                i1, i2 = jidx[j1], jidx[j2]
                frame_data.append(go.Scatter3d(
                    x=[pos[fi, i1, 0], pos[fi, i2, 0]],
                    y=[pos[fi, i1, 1], pos[fi, i2, 1]],
                    z=[pos[fi, i1, 2], pos[fi, i2, 2]]
                ))
            for j in range(len(BODY_JOINTS)):
                frame_data.append(go.Scatter3d(
                    x=[pos[fi, j, 0]], y=[pos[fi, j, 1]], z=[pos[fi, j, 2]]
                ))
            frame_data.append(go.Scatter3d(
                x=[com[fi, 0]], y=[com[fi, 1]], z=[com[fi, 2]]
            ))
            ts = max(0, fi - 300)
            frame_data.append(go.Scatter3d(
                x=com[ts:fi+1, 0].tolist(), 
                y=com[ts:fi+1, 1].tolist(), 
                z=com[ts:fi+1, 2].tolist()
            ))
            anim_frames.append(go.Frame(data=frame_data, name=str(fidx), traces=list(range(len(frame_data)))))
        
        fig.frames = anim_frames
        fps = meta.get('fps', 30)
        dur = 1000 / (fps / step)
        
        fig.update_layout(
            title=dict(text=f'<b>🧗 3D Climbing Skeleton</b><br><sup>{n_frames} frames @ {fps}fps</sup>',
                       x=0.5, font=dict(size=20, color='#fff')),
            scene=dict(
                xaxis=dict(range=[-1.5, 1.5], backgroundcolor='#1a1a2e', gridcolor='#3a3a5e', tickfont=dict(color='#888')),
                yaxis=dict(range=[-1.5, 1.5], backgroundcolor='#1a1a2e', gridcolor='#3a3a5e', tickfont=dict(color='#888')),
                zaxis=dict(range=[-1.5, 1.5], backgroundcolor='#1a1a2e', gridcolor='#3a3a5e', tickfont=dict(color='#888')),
                aspectmode='cube', camera=dict(eye=dict(x=0, y=-2, z=0.3))
            ),
            paper_bgcolor='#0d0d1a', font=dict(color='#fff'),
            margin=dict(l=0, r=0, t=80, b=20), height=700,
            sliders=[{
                'currentvalue': {'prefix': 'Frame: ', 'font': {'color': '#fff'}},
                'len': 0.9, 'x': 0.05, 'bgcolor': '#2a2a4a', 'font': {'color': '#fff'},
                'steps': [{'args': [[str(i)], {'frame': {'duration': 0, 'redraw': True}, 'mode': 'immediate'}],
                           'label': str(indices[i]), 'method': 'animate'} for i in range(len(indices))]
            }],
            updatemenus=[{
                'type': 'buttons', 'x': 0.05, 'y': 0.08,
                'bgcolor': '#2a2a4a', 'bordercolor': '#00D4FF', 'font': {'color': '#fff'},
                'buttons': [
                    {'label': '▶ Play', 'method': 'animate',
                     'args': [None, {'frame': {'duration': dur, 'redraw': True}, 'fromcurrent': True}]},
                    {'label': '⏸', 'method': 'animate', 'args': [[None], {'frame': {'duration': 0}, 'mode': 'immediate'}]},
                ]
            }],
            legend=dict(x=0.98, y=0.98, xanchor='right', bgcolor='rgba(26,26,46,0.8)', font=dict(color='#fff'))
        )
        
        return pio.to_json(fig)
        
    except Exception as e:
        print(f"Error generating Plotly JSON: {e}")
        return None
