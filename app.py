import os
from pathlib import Path

import numpy as np
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from scipy.optimize import least_squares

load_dotenv(Path(__file__).resolve().parent / ".env")

st.set_page_config(page_title="AxisLab | 6-Axis Robotics Tutor", page_icon="🦾", layout="wide")

# Generic illustrative standard-DH model (metres). These dimensions are adjustable in the source.
A = np.array([0.0, 0.30, 0.25, 0.0, 0.0, 0.0])
ALPHA = np.array([np.pi/2, 0.0, 0.0, np.pi/2, -np.pi/2, 0.0])
D = np.array([0.35, 0.0, 0.0, 0.12, 0.10, 0.08])

def dh(a, alpha, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,     ca,    d],
        [0.0,   0.0,    0.0,  1.0]
    ])

def forward_kinematics(q_deg):
    T = np.eye(4)
    points = [T[:3, 3].copy()]
    transforms = []
    for i in range(6):
        T = T @ dh(A[i], ALPHA[i], D[i], np.deg2rad(q_deg[i]))
        transforms.append(T.copy())
        points.append(T[:3, 3].copy())
    return np.array(points), T, transforms

def inverse_position(target, seed):
    def error(q):
        _, T, _ = forward_kinematics(q)
        return T[:3, 3] - target
    result = least_squares(error, np.array(seed, dtype=float),
                           bounds=(-180*np.ones(6), 180*np.ones(6)),
                           max_nfev=250)
    pts, T, _ = forward_kinematics(result.x)
    distance = np.linalg.norm(T[:3, 3] - target)
    return result.x, distance

def make_figure(q, target=None):
    pts, T, transforms = forward_kinematics(q)
    fig = go.Figure()
    fig.add_trace(go.Scatter3d(
        x=pts[:,0], y=pts[:,1], z=pts[:,2],
        mode="lines+markers+text",
        line=dict(width=10, color="#159568"),
        marker=dict(size=[7,8,8,8,8,8,10], color="#153c2e"),
        text=["Base","J1","J2","J3","J4","J5","Tool"],
        textposition="top center",
        name="Robot arm"
    ))
    # Coordinate axes at the tool
    origin = T[:3,3]
    R = T[:3,:3]
    for i, color, name in [(0,"red","Tool X"),(1,"green","Tool Y"),(2,"orange","Tool Z")]:
        end = origin + 0.12*R[:,i]
        fig.add_trace(go.Scatter3d(
            x=[origin[0],end[0]], y=[origin[1],end[1]], z=[origin[2],end[2]],
            mode="lines", line=dict(color=color,width=5), name=name
        ))
    if target is not None:
        fig.add_trace(go.Scatter3d(x=[target[0]],y=[target[1]],z=[target[2]],
                                   mode="markers+text", marker=dict(size=8,color="red",symbol="diamond"),
                                   text=["Target"],textposition="top center",name="Target"))
    fig.update_layout(
        height=570, margin=dict(l=0,r=0,t=30,b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color="#52645a"),
        scene=dict(xaxis_title="X (m)",yaxis_title="Y (m)",zaxis_title="Z (m)",
                   aspectmode="data", bgcolor="#fbfdfb",
                   xaxis=dict(gridcolor="#e4ece7", zerolinecolor="#cbd8d0"),
                   yaxis=dict(gridcolor="#e4ece7", zerolinecolor="#cbd8d0"),
                   zaxis=dict(gridcolor="#e4ece7", zerolinecolor="#cbd8d0")),
        legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="left",x=0)
    )
    return fig, T

if "angles" not in st.session_state:
    st.session_state.angles = [0.0, -35.0, 65.0, 0.0, 35.0, 0.0]

DEFAULT_ANGLES = [0.0, -35.0, 65.0, 0.0, 35.0, 0.0]
PAGES = [
    "Home",
    "Robot Studio",
    "Kinematics Lab",
    "Motion Planning",
    "AI Robotics Tutor",
    "Learning Centre",
    "Project Guide",
    "Settings",
]
PAGE_ICONS = {
    "Home": "home",
    "Robot Studio": "precision_manufacturing",
    "Kinematics Lab": "functions",
    "Motion Planning": "timeline",
    "AI Robotics Tutor": "smart_toy",
    "Learning Centre": "school",
    "Project Guide": "menu_book",
    "Settings": "settings",
}


def reset_pose():
    for i, value in enumerate(DEFAULT_ANGLES, start=1):
        st.session_state[f"j{i}"] = float(value)
    st.session_state.angles = DEFAULT_ANGLES.copy()
    st.session_state.simulation_running = True


def apply_ik_solution():
    solution = st.session_state.get("ik_solution")
    if solution:
        for i, value in enumerate(solution, start=1):
            st.session_state[f"j{i}"] = float(value)
        st.session_state.angles = list(solution)
        st.session_state.page_nav = "Robot Studio"


def navigate_to(page):
    st.session_state.page_nav = page


def start_new_chat():
    st.session_state.conversation_counter += 1
    conversation_id = f"conversation_{st.session_state.conversation_counter}"
    st.session_state.conversations.append(
        {"id": conversation_id, "title": "New conversation", "messages": []}
    )
    st.session_state.active_conversation = conversation_id


def select_conversation(conversation_id):
    st.session_state.active_conversation = conversation_id


def initialize_conversations():
    if "conversations" in st.session_state:
        return
    old_messages = st.session_state.get("tutor_messages", [])
    st.session_state.conversations = []
    st.session_state.conversation_counter = 0
    st.session_state.active_conversation = None
    if old_messages:
        start_new_chat()
        st.session_state.conversations[-1]["messages"] = list(old_messages)
        first_user_message = next(
            (item["content"] for item in old_messages if item.get("role") == "user"),
            "Previous conversation",
        )
        st.session_state.conversations[-1]["title"] = first_user_message[:48]


def current_simulation_context():
    _, pose, _ = forward_kinematics(st.session_state.angles)
    target = st.session_state.get("ik_target")
    solution = st.session_state.get("ik_solution", "none")
    error = st.session_state.get("ik_error", "none")
    dh_rows = [
        [float(A[i]), float(ALPHA[i]), float(D[i]), float(np.deg2rad(st.session_state.angles[i]))]
        for i in range(6)
    ]
    return (
        "Current simulator state (SI units):\n"
        f"Joint angles J1-J6 (degrees): {[round(float(value), 3) for value in st.session_state.angles]}\n"
        f"Standard DH rows [a (m), alpha (rad), d (m), joint theta (rad)]: {dh_rows}\n"
        f"End-effector position XYZ (m): {[round(float(value), 6) for value in pose[:3, 3]]}\n"
        f"End-effector homogeneous transform: {np.round(pose, 6).tolist()}\n"
        f"Last inverse-kinematics target XYZ (m): {target if target is not None else 'none'}\n"
        f"Last IK joint solution: {solution}; position error (m): {error}\n"
        "This is a generic illustrative 6R standard-DH arm. IK solves XYZ position only. "
        "The app has no collision detection, dynamics, hardware connection, or built-in trajectory execution."
    )


def render_robot_figure(q, key, target=None):
    fig, _ = make_figure(q, target)
    st.plotly_chart(fig, width="stretch", key=key)


# Shared page styling. The user requested custom CSS for the fixed navigation and brand treatment.
st.markdown(
    """
    <style>
    :root {
        --axis-ink: #132821;
        --axis-muted: #708078;
        --axis-green: #13865f;
        --axis-border: #e4ece7;
        --axis-soft: #f5f9f6;
    }
    [data-testid="stAppViewContainer"] { background: #f7faf8; }
    [data-testid="stHeader"] { background: rgba(247,250,248,.92); }
    [data-testid="stMain"] .block-container {
        max-width: 1580px; padding-top: 1.2rem; padding-bottom: 5.5rem;
    }
    section[data-testid="stSidebar"] {
        width: 270px !important; min-width: 270px !important;
        border-right: 1px solid #e4ece7;
    }
    section[data-testid="stSidebar"] > div {
        background: linear-gradient(180deg, #f3f9f5 0%, #fbfdfb 55%, #f5faf6 100%);
        border-right: 0;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] > div { gap: .35rem; }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label {
        border-radius: 11px; padding: .42rem .55rem;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {
        background: #e2f3e9; color: #0a6947; font-weight: 650;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        border-color: var(--axis-border) !important; border-radius: 18px !important;
        box-shadow: 0 5px 18px rgba(22, 55, 38, .035);
        background: #fff;
    }
    .axis-brand { display:flex; align-items:center; gap:12px; padding: 8px 4px 18px; }
    .axis-mark {
        width:42px; height:42px; border-radius:14px; display:grid; place-items:center;
        color:#fff; font-weight:800; letter-spacing:-1px; font-size:17px;
        background:linear-gradient(135deg,#147653,#43b879); box-shadow:0 6px 16px #13865f30;
    }
    .axis-brand-name { color:#122b21; font-size:15px; font-weight:760; line-height:1.1; }
    .axis-brand-sub { color:#779087; font-size:10px; letter-spacing:.12em; text-transform:uppercase; margin-top:5px; }
    .axis-topbar { display:flex; align-items:center; justify-content:space-between; padding: 5px 0 22px; }
    .axis-project { font-size:13px; color:#738279; }
    .axis-project strong { color:#253a30; font-weight:650; }
    .axis-profile { display:flex; align-items:center; gap:10px; color:#30453a; font-size:12px; }
    .axis-avatar { display:grid; place-items:center; width:34px; height:34px; border-radius:50%;
        background:linear-gradient(140deg,#c7efd5,#83c9a1); color:#176242; font-weight:750; }
    .axis-hero {
        background:linear-gradient(115deg,#123d2d 0%,#16784f 52%,#4bb87d 100%);
        color:#f5fff8; border-radius:24px; padding:30px 34px; min-height:205px;
        box-shadow:0 16px 34px rgba(23,115,76,.16); position:relative; overflow:hidden;
    }
    .axis-hero:after { content:""; position:absolute; width:310px; height:310px; border-radius:50%;
        right:-90px; top:-145px; border:1px solid #ffffff30; box-shadow:0 0 0 28px #ffffff0d,0 0 0 58px #ffffff08; }
    .axis-eyebrow { text-transform:uppercase; letter-spacing:.13em; color:#bce8cc; font-size:11px; font-weight:700; }
    .axis-hero h1 { color:white; font-size:32px; line-height:1.15; margin:11px 0 8px; letter-spacing:-.03em; }
    .axis-hero p { color:#d8f2e1; margin:0; max-width:620px; font-size:14px; }
    .axis-page-title { color:var(--axis-ink); font-size:27px; font-weight:760; letter-spacing:-.03em; margin:0; }
    .axis-page-subtitle { color:var(--axis-muted); font-size:13px; margin:.35rem 0 1.1rem; }
    .axis-section-label { color:#829187; text-transform:uppercase; letter-spacing:.11em; font-size:10px; font-weight:750; }
    .axis-card-title { color:#1b3729; font-weight:700; font-size:15px; }
    .axis-card-copy { color:#718077; font-size:12px; line-height:1.55; }
    .axis-stat-label { color:#7d8c83; font-size:11px; }
    .axis-stat-value { color:#18382a; font-weight:760; font-size:23px; margin-top:4px; }
    .axis-chat-hint { color:#74857a; background:#f6faf7; border:1px solid #e6eee8; border-radius:14px; padding:12px 14px; font-size:12px; }
    @media (max-width: 900px) {
        section[data-testid="stSidebar"] { min-width: 0 !important; width: min(85vw, 270px) !important; }
        [data-testid="stMain"] .block-container { padding-left:1rem; padding-right:1rem; }
        .axis-hero { padding:24px 22px; }
        .axis-hero h1 { font-size:26px; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "simulation_running" not in st.session_state:
    st.session_state.simulation_running = True
for joint_index, joint_value in enumerate(st.session_state.angles, start=1):
    st.session_state.setdefault(f"j{joint_index}", float(joint_value))
if "page_nav" not in st.session_state:
    st.session_state.page_nav = "Home"
initialize_conversations()

# Persistent application navigation and project identity.
with st.sidebar:
    st.markdown(
        '<div class="axis-brand"><div class="axis-mark">AX</div><div>'
        '<div class="axis-brand-name">AxisLab</div><div class="axis-brand-sub">Robotics learning</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="axis-section-label">Workspace</div>', unsafe_allow_html=True)
    page = st.radio(
        "Workspace navigation",
        PAGES,
        format_func=lambda item: f":material/{PAGE_ICONS[item]}:  {item}",
        label_visibility="collapsed",
        key="page_nav",
    )
    st.write("")
    st.markdown(
        '<div class="axis-chat-hint"><b>GENERIC 6R MODEL</b><br>'
        'Educational simulation · Pose saved for this session.</div>',
        unsafe_allow_html=True,
    )
# Common top bar.
st.markdown(
    '<div class="axis-topbar"><div class="axis-project"><strong>AxisLab</strong> &nbsp; / &nbsp; 6-Axis Robotics Arm Tutor</div>'
    '<div class="axis-profile"><span>Robotics learner</span><span class="axis-avatar">RL</span></div></div>',
    unsafe_allow_html=True,
)


def page_heading(title, subtitle):
    st.markdown(
        f'<div class="axis-page-title">{title}</div><div class="axis-page-subtitle">{subtitle}</div>',
        unsafe_allow_html=True,
    )


def mini_stat(label, value, detail):
    st.markdown(
        f'<div class="axis-stat-label">{label}</div><div class="axis-stat-value">{value}</div>'
        f'<div class="axis-card-copy">{detail}</div>',
        unsafe_allow_html=True,
    )


if page == "Home":
    _, home_pose, _ = forward_kinematics(st.session_state.angles)
    st.markdown(
        '<div class="axis-hero"><div class="axis-eyebrow">Interactive robotics workspace</div>'
        '<h1>Build intuition for<br>robot motion.</h1>'
        '<p>Explore a six-axis robot, experiment with kinematics, and learn from your live simulation.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    overview = st.columns(4)
    with overview[0].container(border=True):
        mini_stat("ROBOT MODEL", "6-axis", "Six rotary joints · generic 6R")
    with overview[1].container(border=True):
        mini_stat("TOOL X", f"{home_pose[0, 3]:.3f} m", "Live end-effector position")
    with overview[2].container(border=True):
        mini_stat("TOOL Y", f"{home_pose[1, 3]:.3f} m", "Live end-effector position")
    with overview[3].container(border=True):
        mini_stat("TOOL Z", f"{home_pose[2, 3]:.3f} m", "Live end-effector position")

    st.write("")
    left, right = st.columns([1.2, 1], gap="large")
    with left.container(border=True):
        st.markdown('<div class="axis-section-label">Continue exploring</div>', unsafe_allow_html=True)
        st.subheader("Robot Studio")
        st.caption("Adjust all six joints and inspect the live 3D pose and tool coordinates.")
        st.button("Open Robot Studio", icon=":material/arrow_forward:", type="primary", on_click=navigate_to, args=("Robot Studio",), key="home_studio")
    with right.container(border=True):
        st.markdown('<div class="axis-section-label">Learn by doing</div>', unsafe_allow_html=True)
        st.subheader("Kinematics Lab")
        st.caption("Solve for joint angles that bring the tool close to an XYZ target.")
        st.button("Open Kinematics Lab", icon=":material/arrow_forward:", on_click=navigate_to, args=("Kinematics Lab",), key="home_kinematics")

    st.write("")
    st.markdown('<div class="axis-section-label">Your workspace</div>', unsafe_allow_html=True)
    quick = st.columns(3, gap="medium")
    with quick[0].container(border=True):
        st.markdown("**Motion planning**")
        st.caption("Preview joint-space interpolation with linear or cubic timing.")
        st.button("Plan a motion", key="home_motion", on_click=navigate_to, args=("Motion Planning",))
    with quick[1].container(border=True):
        st.markdown("**AI robotics tutor**")
        st.caption("Ask about the robot and get answers grounded in its current pose.")
        st.button("Ask a question", key="home_tutor", on_click=navigate_to, args=("AI Robotics Tutor",))
    with quick[2].container(border=True):
        st.markdown("**Learning centre**")
        st.caption("Review DH parameters, coordinate frames, and workspace concepts.")
        st.button("Explore lessons", key="home_learning", on_click=navigate_to, args=("Learning Centre",))
    st.caption("Educational simulator only · No hardware connection or certified safety functions.")

elif page == "Robot Studio":
    page_heading("Robot Studio", "Adjust the six joints and inspect the resulting tool pose in real time.")
    control_col, scene_col = st.columns([0.9, 1.65], gap="large")
    with control_col.container(border=True):
        head, toggle = st.columns([1.2, 1], vertical_alignment="center")
        with head:
            st.markdown("**Joint controls**")
        with toggle:
            st.toggle("Simulation running", key="simulation_running", help="Pause to freeze joint controls.")
        for i in range(6):
            st.slider(
                f"Joint J{i + 1} · degrees",
                -180.0,
                180.0,
                step=1.0,
                key=f"j{i + 1}",
                disabled=not st.session_state.simulation_running,
            )
        st.button("Reset to home pose", icon=":material/restart_alt:", on_click=reset_pose, width="stretch")
    angles = [float(st.session_state[f"j{i}"]) for i in range(1, 7)]
    st.session_state.angles = angles
    points, pose, _ = forward_kinematics(angles)
    with scene_col.container(border=True):
        st.markdown("**Live robot simulation**")
        st.caption("Drag to orbit · Scroll to zoom · Axes are in metres")
        render_robot_figure(angles, "studio_robot")
    values = st.columns(4)
    for col, label, value, detail in zip(
        values,
        ["End-effector X", "End-effector Y", "End-effector Z", "Joint state"],
        [f"{pose[0, 3]:.3f} m", f"{pose[1, 3]:.3f} m", f"{pose[2, 3]:.3f} m", "6 DOF"],
        ["World frame", "World frame", "World frame", "All revolute"],
    ):
        with col.container(border=True):
            mini_stat(label, value, detail)
    with st.expander("Current joint angles and DH model"):
        st.dataframe(
            {"Joint": [f"J{i}" for i in range(1, 7)], "Angle (°)": [round(x, 2) for x in angles],
             "a (m)": A, "alpha (rad)": ALPHA, "d (m)": D},
            hide_index=True,
            width="stretch",
        )

elif page == "Kinematics Lab":
    page_heading("Kinematics Lab", "Use inverse kinematics to find a joint configuration for a target tool position.")
    left, right = st.columns([0.95, 1.5], gap="large")
    _, current_pose, _ = forward_kinematics(st.session_state.angles)
    with left.container(border=True):
        st.markdown("**Target position**")
        st.caption("Target fields default to the tool position when you first open this lab.")
        st.session_state.setdefault("ik_tx", float(round(current_pose[0, 3], 3)))
        st.session_state.setdefault("ik_ty", float(round(current_pose[1, 3], 3)))
        st.session_state.setdefault("ik_tz", float(round(current_pose[2, 3], 3)))
        tx = st.number_input("Target X · metres", step=0.01, format="%.3f", key="ik_tx")
        ty = st.number_input("Target Y · metres", step=0.01, format="%.3f", key="ik_ty")
        tz = st.number_input("Target Z · metres", step=0.01, format="%.3f", key="ik_tz")
        if st.button("Solve inverse kinematics", type="primary", icon=":material/calculate:", width="stretch"):
            target = np.array([tx, ty, tz], dtype=float)
            q_sol, err = inverse_position(target, st.session_state.angles)
            st.session_state.ik_solution = q_sol.tolist()
            st.session_state.ik_error = float(err)
            st.session_state.ik_target = target.tolist()
        st.caption("The numerical solver minimizes XYZ position error. Tool orientation is not constrained.")
        if "ik_solution" in st.session_state:
            st.metric("Position error", f"{st.session_state.ik_error * 1000:.2f} mm")
            st.button("Apply solution to Robot Studio", icon=":material/check:", on_click=apply_ik_solution, width="stretch")
    with right.container(border=True):
        st.markdown("**Solution preview**")
        if "ik_solution" in st.session_state:
            target = np.array([tx, ty, tz], dtype=float)
            render_robot_figure(st.session_state.ik_solution, "ik_robot", target)
            st.dataframe(
                {"Joint": [f"J{i}" for i in range(1, 7)], "Solution angle (°)": [round(x, 2) for x in st.session_state.ik_solution]},
                hide_index=True,
                width="stretch",
            )
        else:
            render_robot_figure(st.session_state.angles, "ik_robot_current")
            st.info("Calculate a solution to compare the target with the reachable tool position.")

elif page == "Motion Planning":
    page_heading("Motion Planning", "Preview a joint-space move from the current pose to a selected goal configuration.")
    setup, preview = st.columns([0.9, 1.55], gap="large")
    with setup.container(border=True):
        st.markdown("**Trajectory setup**")
        st.caption("Choose a goal angle for each revolute joint.")
        target_angles = []
        for i in range(6):
            goal_key = f"motion_goal_{i + 1}"
            st.session_state.setdefault(goal_key, DEFAULT_ANGLES[i])
            target_angles.append(
                st.number_input(
                    f"Goal J{i + 1} · degrees", -180.0, 180.0, step=5.0,
                    key=goal_key,
                )
            )
        duration = st.slider("Move duration · seconds", 1.0, 10.0, 3.0, 0.5, key="motion_duration")
        profile = st.selectbox("Timing profile", ["Cubic ease", "Linear"], key="motion_profile")
        progress = st.slider("Preview position · percent", 0, 100, 50, key="motion_progress")
    start_angles = np.asarray(st.session_state.angles, dtype=float)
    goal_angles = np.asarray(target_angles, dtype=float)
    sample_count = 60
    normalized_time = np.linspace(0.0, 1.0, sample_count)
    blend = 3 * normalized_time**2 - 2 * normalized_time**3 if profile == "Cubic ease" else normalized_time
    trajectory = start_angles[None, :] + blend[:, None] * (goal_angles - start_angles)[None, :]
    tool_path = np.array([forward_kinematics(q)[1][:3, 3] for q in trajectory])
    current_blend = progress / 100.0
    if profile == "Cubic ease":
        current_blend = 3 * current_blend**2 - 2 * current_blend**3
    preview_angles = start_angles + current_blend * (goal_angles - start_angles)
    with preview.container(border=True):
        st.markdown("**Planned path preview**")
        st.caption(f"{duration:.1f} s · {profile.lower()} timing · {sample_count} samples")
        motion_fig, _ = make_figure(preview_angles)
        motion_fig.add_trace(
            go.Scatter3d(
                x=tool_path[:, 0], y=tool_path[:, 1], z=tool_path[:, 2],
                mode="lines", line=dict(color="#29a36a", width=7, dash="dash"),
                name="Tool path",
            )
        )
        motion_fig.update_layout(height=440)
        st.plotly_chart(motion_fig, width="stretch", key="motion_preview")
        joint_fig = go.Figure()
        time_values = normalized_time * duration
        for i in range(6):
            joint_fig.add_trace(go.Scatter(x=time_values, y=trajectory[:, i], mode="lines", name=f"J{i + 1}"))
        joint_fig.update_layout(
            height=255, margin=dict(l=10, r=10, t=15, b=10),
            xaxis_title="Time (s)", yaxis_title="Joint angle (°)",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=1.15),
        )
        st.plotly_chart(joint_fig, width="stretch", key="motion_joint_curves")
    st.info("This page previews mathematical joint interpolation only. It does not execute the path or check collisions, velocity, acceleration, or hardware limits.")

elif page == "AI Robotics Tutor":
    page_heading("AI Robotics Tutor", "Ask engineering questions grounded in the robot model and the live simulation.")
    initialize_conversations()
    history_col, chat_col = st.columns([0.78, 1.9], gap="large")
    with history_col.container(border=True):
        st.markdown("**Conversations**")
        if st.button("New chat", icon=":material/add:", type="primary", width="stretch", on_click=start_new_chat):
            pass
        search_text = st.text_input("Search conversations", placeholder="Search your history", icon=":material/search:", key="conversation_search")
        matching = [
            item for item in st.session_state.conversations
            if search_text.casefold() in item["title"].casefold()
        ]
        if not matching:
            st.caption("No matching conversations yet. Start a new chat to begin.")
        for item in reversed(matching):
            label = item["title"] or "New conversation"
            if st.button(
                label[:42],
                key=f"select_{item['id']}",
                type="primary" if item["id"] == st.session_state.active_conversation else "secondary",
                width="stretch",
            ):
                select_conversation(item["id"])
                st.rerun()
    active_conversation = next(
        (item for item in st.session_state.conversations if item["id"] == st.session_state.active_conversation),
        None,
    )
    if active_conversation is None:
        start_new_chat()
        active_conversation = st.session_state.conversations[-1]
    with chat_col.container(border=True):
        st.markdown(f"**{active_conversation['title']}**")
        st.caption("AI tutor · live 6-axis simulation context")
        messages = active_conversation["messages"]
        if not messages:
            st.markdown(
                '<div class="axis-chat-hint">Start with a question such as “How do these joint angles affect the tool position?” or “Explain the DH rows in this model.”</div>',
                unsafe_allow_html=True,
            )
        for message in messages:
            role = message["role"]
            avatar = ":material/person:" if role == "user" else ":material/smart_toy:"
            with st.chat_message(role, avatar=avatar):
                st.markdown(message["content"])
    prompt = st.chat_input("Ask about kinematics, your current pose, or robot motion…", key="tutor_chat_input")
    if prompt:
        messages = active_conversation["messages"]
        messages.append({"role": "user", "content": prompt})
        if active_conversation["title"] == "New conversation":
            active_conversation["title"] = prompt[:48]
        with chat_col:
            with st.chat_message("user", avatar=":material/person:"):
                st.markdown(prompt)
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            st.error("AI Tutor is unavailable: GROQ_API_KEY is missing. Add it to the app's .env file and restart Streamlit.")
        else:
            try:
                from groq import (
                    APIConnectionError,
                    APIStatusError,
                    APITimeoutError,
                    AuthenticationError,
                    BadRequestError,
                    Groq,
                    NotFoundError,
                    RateLimitError,
                )
            except ImportError:
                st.error("The Groq SDK is not installed. Run `pip install -r requirements.txt` in the project environment.")
            else:
                try:
                    system_prompt = (
                        "You are a knowledgeable, patient robotics and mechanical engineering tutor for this "
                        "Python 6-axis robot simulator. Answer the user's actual question and explain step by step "
                        "when helpful. Teach forward and inverse kinematics, standard Denavit-Hartenberg parameters, "
                        "joint angles, homogeneous transformations, workspace and reachability, trajectory planning, "
                        "Python code, and simulation results. Use equations or small numeric examples when useful. "
                        "Use the provided live simulation values for questions about the current robot. Clearly say "
                        "when the app does not calculate something; do not invent results or imply this generic model "
                        "represents a specific physical robot.\n\n"
                        + current_simulation_context()
                    )
                    model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
                    api_messages = [{"role": "system", "content": system_prompt}, *messages]
                    with chat_col:
                        with st.chat_message("assistant", avatar=":material/smart_toy:"):
                            with st.spinner("Working through the robot model…"):
                                response = Groq(api_key=api_key).chat.completions.create(
                                    model=model,
                                    messages=api_messages,
                                    max_tokens=900,
                                )
                            answer = response.choices[0].message.content
                            if not answer or not answer.strip():
                                st.error("Groq returned an empty answer. Please try again.")
                            else:
                                st.markdown(answer)
                                messages.append({"role": "assistant", "content": answer})
                except RateLimitError as exc:
                    st.error(f"Groq rate limit reached. Wait briefly and try again. Details: {exc}")
                except (APITimeoutError, APIConnectionError) as exc:
                    st.error(f"Could not connect to Groq. Check your network connection and try again. Details: {exc}")
                except AuthenticationError:
                    st.error("Groq rejected GROQ_API_KEY. Check the key in your .env file and restart Streamlit.")
                except (NotFoundError, BadRequestError) as exc:
                    error_text = str(exc)
                    if isinstance(exc, NotFoundError) or "model" in error_text.casefold():
                        st.error(f"Groq could not find or use model '{model}'. Check the GROQ_MODEL value. Details: {exc}")
                    else:
                        st.error(f"Groq rejected the request. Check the request settings. Details: {exc}")
                except APIStatusError as exc:
                    st.error(f"Groq API returned HTTP {exc.status_code}. Please retry or check your Groq account. Details: {exc}")
                except Exception as exc:
                    st.error(f"Groq Tutor request failed: {exc}")

elif page == "Learning Centre":
    page_heading("Learning Centre", "Short guided modules for understanding the mechanics and mathematics of robot arms.")
    lesson_columns = st.columns(3, gap="large")
    lessons = [
        ("01 · Robot geometry", "Frames and DH parameters", "A standard DH row describes the transform between adjacent joint frames with a, alpha, d, and theta. This app uses metres and radians for geometry; slider angles are converted from degrees."),
        ("02 · Kinematics", "From joint angles to tool pose", "Forward kinematics multiplies the six homogeneous transforms: T₀₆ = T₀₁ T₁₂ … T₅₆. The final column of T₀₆ gives the tool position in the base frame."),
        ("03 · Reachability", "What can this arm reach?", "Reachability depends on link dimensions, joint limits, and configuration. The numerical IK solver searches from the current pose and targets XYZ position only; a low residual is evidence of a close positional solution."),
        ("04 · Motion", "Joint-space trajectories", "A joint-space trajectory interpolates each joint from a start angle to a goal. Cubic easing makes the interpolation rate zero at both ends, but this educational preview does not model dynamics or enforce speed limits."),
        ("05 · Simulation", "Read the 3D result", "The plotted tool coordinate axes show the orientation of the final frame. The position values use metres. Changing a joint changes the transform chain and updates the displayed pose."),
        ("06 · Python", "Explore the implementation", "The DH function returns a 4×4 transform. forward_kinematics multiplies one transform per joint. inverse_position uses SciPy least_squares to minimize the difference between target and computed XYZ."),
    ]
    for index, lesson in enumerate(lessons):
        with lesson_columns[index % 3].container(border=True):
            st.markdown(f'<div class="axis-section-label">{lesson[0]}</div>', unsafe_allow_html=True)
            st.subheader(lesson[1])
            st.write(lesson[2])
    st.write("")
    st.button("Open AI robotics tutor", icon=":material/arrow_forward:", on_click=navigate_to, args=("AI Robotics Tutor",))

elif page == "Project Guide":
    page_heading("Project Guide", "How the simulator is built, what it models, and where its current limits are.")
    guide_left, guide_right = st.columns([1.1, 0.9], gap="large")
    with guide_left.container(border=True):
        st.markdown("**Project objective**")
        st.write("An educational Python application that simulates a generic six-revolute-joint robot and teaches core industrial robotics concepts.")
        st.markdown("**Core modules**")
        st.markdown(
            "- **Simulation engine:** standard Denavit–Hartenberg homogeneous transformations.\n"
            "- **Forward kinematics:** computes tool pose from six joint angles.\n"
            "- **Inverse kinematics:** numerical least-squares optimization for a requested XYZ position.\n"
            "- **Visualization:** interactive Plotly 3D model with tool coordinate axes.\n"
            "- **AI assistance:** Groq API using `GROQ_API_KEY` and optional `GROQ_MODEL`."
        )
    with guide_right.container(border=True):
        st.markdown("**Model limitations**")
        st.markdown(
            "- Link dimensions and DH parameters are illustrative, not from a manufacturer.\n"
            "- IK targets position only, not full tool orientation.\n"
            "- No collision detection, dynamics, payload calculations, or hardware control.\n"
            "- Motion planning is a joint interpolation preview, not a robot controller."
        )
        st.warning("For education and simulation only. This app is not a certified industrial controller.")
    with st.expander("Suggested demonstration"):
        st.markdown("1. Move J1–J6 in Robot Studio and observe the 3D pose.\n2. Read the live XYZ tool position.\n3. Use Kinematics Lab to solve for a nearby target.\n4. Preview a joint-space move in Motion Planning.\n5. Ask the tutor to explain the DH transform chain.")

elif page == "Settings":
    page_heading("Settings", "Application configuration and model status for this session.")
    settings_col, status_col = st.columns(2, gap="large")
    with settings_col.container(border=True):
        st.markdown("**Tutor configuration**")
        configured_model = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile").strip() or "llama-3.3-70b-versatile"
        st.text_input("Groq model", value=configured_model, disabled=True, help="Configure with the GROQ_MODEL environment variable in .env.")
        key_is_set = bool(os.getenv("GROQ_API_KEY", "").strip())
        st.write("API key status")
        if key_is_set:
            st.success("GROQ_API_KEY is configured")
        else:
            st.warning("GROQ_API_KEY is not configured")
        st.caption("The app loads GROQ_API_KEY from the project .env file and never displays the key.")
    with status_col.container(border=True):
        st.markdown("**Robot model**")
        st.write("Generic 6R arm · standard DH convention · metres and radians in the transform model")
        st.dataframe(
            {"Joint": [f"J{i}" for i in range(1, 7)], "a (m)": A, "alpha (rad)": ALPHA, "d (m)": D},
            hide_index=True,
            width="stretch",
        )
        st.caption("Joint angles are displayed in degrees. The model is illustrative and has no hardware connection.")
