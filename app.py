import streamlit as st
from ultralytics import YOLO
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.ops import nms
import os
import pandas as pd
import json
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = []


st.set_page_config(
    page_title="OceanGuard AI",
    page_icon="🌊",
    layout="wide"
)

st.markdown("""
<style>
body, .stApp { background: #e9f4fb; font-family: 'Inter', sans-serif; }

/* ------------------- CINEMATIC GRADIENT BACKGROUND STYLES ------------------- */
.hero-video-box { 
    position: relative;
    width: 100%;
    height: 330px; 
    border-radius: 18px;
    margin-bottom: 35px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.55);
    background: linear-gradient(135deg, 
                #001a33 0%, 
                #003366 50%, 
                #004d99 100% 
                );
    filter: brightness(0.8) saturate(1.5) contrast(1.2); 
}
.hero-text-overlay {
    position: absolute; 
    z-index: 99; 
    top: 165px; 
    left: 50%; 
    transform: translate(-50%, -50%); 
    text-align: center; 
    color: white; 
    width: 100%;
    pointer-events: none;
}
.hero-text-overlay h1 { font-size: 75px; font-weight: 900; color: #53c7ff; text-shadow: 3px 3px 20px rgba(0, 0, 0, 1); }
.hero-text-overlay p { font-size: 23px; opacity: 0.92; text-shadow: 1px 1px 6px rgba(0, 0, 0, 1); }

/* GLASS CARD */
.glass-card { background: rgba(255,255,255,0.55); border-radius:18px; padding:22px; backdrop-filter: blur(16px) saturate(160%); -webkit-backdrop-filter: blur(16px) saturate(160%); border:1px solid rgba(255,255,255,0.4); box-shadow:0px 8px 24px rgba(0,0,0,0.12); margin-bottom:20px; }
.metric-value { font-size:4rem; font-weight:900; color:#105e99; }
.metric-label { font-size:1.3rem; color:#444; }

/* Modern Table */
.analysis-table { width:100%; border-collapse:collapse; font-size:1rem; margin-top:15px; }
.analysis-table th { background:#f2f8ff; color:#0b518a; padding:12px; text-align:left; font-weight:700; border-bottom:2px solid #dce9f7; }
.analysis-table td { padding:10px; border-bottom:1px solid #ecf2f8; }
.analysis-table tr:hover td { background:#f9fcff; }
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<div class="hero-video-box">
    <div class="hero-text-overlay">
        <h1>OceanGuard AI</h1>
        <p>Premium Visual Intelligence for Underwater Environmental Assessment</p>
        <p style="font-size:15px; opacity:0.7;">Dual-Model Detection • Ecological Analytics • Mapped Results</p>
    </div>
</div>
""", unsafe_allow_html=True)


def load_yolo_model(path):
    if os.path.exists(path):
        try: return YOLO(path)
        except: return None
    return None

@st.cache_resource(ttl=3600)
def load_models():
    return load_yolo_model("models/underwater_plastics_model.pt"), load_yolo_model("models/algae_detection_model.pt")

underwater_model, algae_model = load_models()
underwater_names = underwater_model.names if underwater_model else {}
algae_names = algae_model.names if algae_model else {}

pollution_weights = {
    "plastic":0.9,"pbag":0.85,"pbottle":0.8,"net":0.7,"tire":0.65,"Mask":0.6,
    "glove":0.5,"can":0.4,"gbottle":0.35,"electronics":0.3,"cellphone":0.3,
    "metal":0.25,"misc":0.15,"rod":0.1,"sunglasses":0.05,
    "Haematococcus":0.6,"Porphyridium":0.55,"Dunaliella salina":0.4,
    "Effrenium":0.35,"Chlorella":0.2,"Platymonas":0.15
}
default_weight = 0.1
max_possible_impact = max(pollution_weights.values()) if pollution_weights else 1.0

with st.sidebar:
    st.subheader("📍 Location Input")
    
    loc_name = st.text_input("Location Name/ID", value="Sample Site A")
    loc_lat = st.number_input("Latitude", value=22.302711, format="%.6f")
    loc_lon = st.number_input("Longitude", value=114.177216, format="%.6f")
    
    st.markdown("---")
    st.subheader("⚙️ Model Settings")
    run_mode = st.selectbox("Detection Mode", ["Both (merge)","Underwater debris only","Algae only"])
    conf_thresh = st.slider("Confidence Threshold",0.05,0.9,0.35)
    iou_thresh = st.slider("Merge IoU Threshold",0.1,0.9,0.45)
    st.markdown("---")
    if underwater_model: st.success(f"Debris Model Loaded ({len(underwater_names)} classes)")
    else: st.error("Debris Model Missing")
    if algae_model: st.success(f"Algae Model Loaded ({len(algae_names)} classes)")
    else: st.error("Algae Model Missing")

st.markdown("<h3>📤 Upload Image for Analysis</h3>", unsafe_allow_html=True)
uploaded = st.file_uploader("Upload underwater image (jpg/png)", type=["jpg","jpeg","png"])


def result_to_boxes(res, class_names, source):
    boxes=[]
    if not res or len(res)==0: return boxes
    r = res[0]
    if not hasattr(r,'boxes') or r.boxes is None: return boxes
    
    xyxy = r.boxes.xyxy.cpu().numpy()
    conf = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy().astype(int)
    
    for b,c,cl in zip(xyxy,conf,cls):
        boxes.append({
            "xyxy":b,
            "conf":float(c),
            "cls":int(cl),
            "label": class_names.get(int(cl), f"{source}_{cl}"),
            "source": source
        })
    return boxes

def merge_all_boxes(all_boxes, iou_thresh=0.45):
    if not all_boxes: return []
    boxes = torch.tensor([b["xyxy"] for b in all_boxes],dtype=torch.float32)
    scores = torch.tensor([b["conf"] for b in all_boxes],dtype=torch.float32)
    keep = nms(boxes,scores,iou_thresh).cpu().numpy().tolist()
    return [all_boxes[i] for i in keep]


def draw_boxes_pil(img, boxes):
    draw = ImageDraw.Draw(img)
    
    
    try: 
        FONT_SIZE = 30
        font = ImageFont.truetype("Arial Bold.ttf", FONT_SIZE)
    except: 
        FONT_SIZE = 30
        try: font = ImageFont.truetype("arial.ttf", FONT_SIZE) 
        except: font = ImageFont.load_default()
    
    BOX_LINE_WIDTH = 7
    
    
    for b in boxes:
        x1,y1,x2,y2 = map(int,b["xyxy"])
        txt = f"{b.get('label','').upper()} {b['conf']:.2f}"
        
        color="#973410" if b.get("source","")=="underwater" else "#3CB371" 
        
        draw.rectangle([x1,y1,x2,y2], outline=color, width=BOX_LINE_WIDTH)
        
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        
        if hasattr(temp_draw, 'textbbox'):
             tw, th = temp_draw.textbbox((0, 0), txt, font=font)[2:]
        else:
             tw, th = temp_draw.textsize(txt, font=font) 
        
        text_padding_y = 16
        text_padding_x = 17
        
        draw.rectangle([x1, y1 - th - text_padding_y, x1 + tw + text_padding_x, y1], fill=color)
        draw.text((x1 + text_padding_x // 2, y1 - th - (text_padding_y - (text_padding_y // 4))), txt, fill="white", font=font)
        
    return img


if uploaded:
    img = Image.open(uploaded).convert("RGB")
    img_np = np.array(img)

    st.markdown("### 🔍 Running Detection...")
    all_boxes=[]

    if run_mode in ("Both (merge)","Underwater debris only") and underwater_model:
        if underwater_model:
            res_u = underwater_model.predict(source=img_np, conf=conf_thresh, device='cpu')
            all_boxes += result_to_boxes(res_u, underwater_names, "underwater")

    if run_mode in ("Both (merge)","Algae only") and algae_model:
        if algae_model:
            res_a = algae_model.predict(source=img_np, conf=conf_thresh, device='cpu')
            all_boxes += result_to_boxes(res_a, algae_names, "algae")

    merged = merge_all_boxes(all_boxes, iou_thresh) if run_mode=="Both (merge)" else all_boxes

   
    pollution_pct = 0.0
    if merged:
        weights = np.array([pollution_weights.get(b["label"], default_weight) for b in merged])
        confs = np.array([b["conf"] for b in merged])
        
        detection_scores = confs * weights
        
        if len(merged) > 0 and max_possible_impact > 0:
            raw_index = detection_scores.sum() / len(merged)
            pollution_pct = (raw_index / max_possible_impact) * 100
        
        pollution_pct = max(0.0, min(100.0, pollution_pct)) 

    
    if pollution_pct < 20:
        severity=("Low","Excellent","✅","#28a745","Water is healthy with minimal contaminants, indicating an **Excellent** ecological status. Regular monitoring is sufficient.")
        map_color = [0, 255, 0] # Green
    elif pollution_pct < 45:
        severity=("Moderate","Acceptable","⚠️","#ffc107","Moderate indicators detected. Further monitoring and a targeted cleanup/investigation plan are advised for an **Acceptable** ecological status.")
        map_color = [255, 255, 0] # Yellow
    elif pollution_pct < 75:
        severity=("High","Concerning","🟠","#fd7e14","Significant pollutants or algae present. **Action is required** to address this **Concerning** status, including source tracing and immediate mitigation.")
        map_color = [255, 140, 0] # Orange
    else:
        severity=("Critical","Unsafe","🔴","#dc3545","Severe contamination risk detected. **Immediate intervention** and resource deployment are necessary for this **Unsafe** status to prevent ecological harm.")
        map_color = [255, 0, 0] # Red

    vis = draw_boxes_pil(img.copy(), merged)
    
    
    st.session_state.analysis_data.append({
        'location': loc_name,
        'lat': loc_lat,
        'lon': loc_lon,
        'pollution_index': pollution_pct,
        'severity': severity[1],
        'color': map_color,
        'count': len(merged)
    })
    
    
    col1, col2 = st.columns([6, 4]) 
    
    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🖼️ Detection Visualization (YOLO Type)")
        st.image(vis, use_container_width=True, caption=f"Total Objects Detected: {len(merged)}")
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("📊 Ecosystem Summary")
        
        st.markdown(f"**Location:** {loc_name} ({loc_lat:.2f}, {loc_lon:.2f})")
        st.markdown("---")
        
        # Pollution Index
        st.markdown(f"""
            <div style="text-align:center;">
                <p class="metric-value">{pollution_pct:.1f}%</p>
                <p class="metric-label">Pollution Index</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Water Status
        st.markdown(f"""
            <div style="
                margin-top:20px;
                padding:15px;
                border-radius:15px;
                border:2px solid {severity[3]};
                background:{severity[3]}22;
                text-align:center;">
                <h3 style="color:{severity[3]}; margin:0;">{severity[2]} STATUS: {severity[1]}</h3>
                <p style="font-size:1.0rem; color:{severity[3]}; margin:0;">Risk Level: {severity[0]}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Advice/Summary
        st.markdown("---")
        st.markdown(f"**Environmental Advisory:** {severity[4]}")
        
        st.markdown('</div>', unsafe_allow_html=True)


    # ------------------- Analytical table (Detailed Breakdown) -------------------
    df_results = pd.DataFrame([{
        "Label": b["label"].upper(),
        "Source": "DEBRIS" if b["source"]=="underwater" else "ALGAE",
        "Confidence": b["conf"],
        "Impact": pollution_weights.get(b["label"], default_weight)
    } for b in merged])

    if not df_results.empty:
        df_grouped = df_results.groupby("Label").agg(
            Count=("Label","size"),
            Avg_Confidence=("Confidence","mean"),
            Max_Impact=("Impact","max"),
            Source=("Source","first")
        ).reset_index()

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("🧪 Analytical Breakdown (Counts & Impact)")
        table_html = "<table class='analysis-table'><thead><tr><th>Object</th><th>Source</th><th>Count</th><th>Avg Conf.</th><th>Max Impact</th></tr></thead><tbody>"
        for _,r in df_grouped.iterrows():
            table_html += f"<tr><td><b>{r['Label']}</b></td><td>{r['Source']}</td><td>{r['Count']}</td><td>{r['Avg_Confidence']:.2f}</td><td>{r['Max_Impact']:.2f}</td></tr>"
        table_html += "</tbody></table>"
        st.markdown(table_html, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
    st.success(f"Data for {loc_name} has been added to the map!")

# ------------------- MAP VISUALIZATION SECTION -------------------
if st.session_state.analysis_data:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("📍 Environmental Monitoring Map (Hot/Cold Spots)")
    
    # Convert stored session state data to a DataFrame for mapping
    map_df = pd.DataFrame(st.session_state.analysis_data)
    
    # Prepare data for Streamlit's st.map (requires 'lat' and 'lon')
    map_data = map_df[['lat', 'lon']].copy()
    
    # Add optional size and color columns for better visualization (requires a custom library like pydeck/altair, but we'll use st.map for simplicity first)
    # Streamlit's native st.map is simple and doesn't support color-coding easily.
    # To demonstrate the color/hotspot concept, we'll use st.columns with a warning.
    
    
    # --- STEP 2 & 3: DISPLAY THE MAP ---
    st.map(map_data, zoom=10)
    
    st.markdown("---")
    st.markdown(f"**Total Sample Sites:** {len(st.session_state.analysis_data)}")
    
    # Show the results table for quick verification
    st.dataframe(
        map_df[['location', 'lat', 'lon', 'pollution_index', 'severity']],
        column_config={
            "pollution_index": st.column_config.ProgressColumn(
                "Pollution Index (%)",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
            "location": "Site Name",
            "severity": "Status",
            "lat": "Latitude",
            "lon": "Longitude",
        },
        hide_index=True,
        use_container_width=True
    )
    

    
    st.markdown('</div>', unsafe_allow_html=True)


# ------------------- FOOTER -------------------
st.markdown("""
<div style="text-align:center; color:#777; margin-top:40px;">
© 2025 AquaSense AI • AI-driven underwater environmental analytics.
</div>
""", unsafe_allow_html=True)