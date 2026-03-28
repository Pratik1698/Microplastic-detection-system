import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN
from PIL import Image
from ultralytics import YOLO

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="EcoTrack AI | Marine Microplastic System", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# --- 2. ASSET & MODEL LOADING ---

@st.cache_resource
def load_yolo_model():
    """Load the custom-trained YOLOv8 Microplastic detector."""
    try:
        return YOLO('best.pt')
    except Exception as e:
        st.error(f"Error loading best.pt: {e}")
        return None

@st.cache_data
def load_and_clean_raman():
    """Load and prepare Raman Spectroscopy dataset."""
    df = pd.read_csv('final_processed_microplastics_17k.csv')
    polymers = ['PE', 'PS', 'PMMA', 'PTFE', 'NYLON']
    
    def clean_label(label):
        label_upper = str(label).upper()
        found = [p for p in polymers if p in label_upper]
        return "_".join(sorted(found)) if found else 'BACKGROUND'
    
    df['clean_category'] = df['category'].apply(clean_label)
    return df

@st.cache_resource
def train_raman_model(df):
    """Train Random Forest on Raman spectral data."""
    X = df.drop(columns=['category', 'source_file', 'clean_category'])
    y = df['clean_category']
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X, y)
    return model, X.columns

@st.cache_data
def generate_hotspot_map():
    """Combine global datasets and cluster hotspots using DBSCAN."""
    try:
        df_adv = pd.read_csv('ADVENTURE_MICRO_FROM_SCIENTIST.csv')
        df_geo = pd.read_csv('GEOMARINE_MICRO.csv')
        df_sea = pd.read_csv('SEA_MICRO.csv')
        
        df_adv_clean = df_adv[df_adv['Total_Pieces_L'] > 0][['Latitude', 'Longitude']]
        df_geo_clean = df_geo[df_geo['MP_conc__particles_cubic_metre_'] > 0][['Latitude', 'Longitude']]
        df_sea_clean = df_sea[df_sea['Pieces_KM2'] > 0][['Latitude', 'Longitude']]
        
        df_all = pd.concat([df_adv_clean, df_geo_clean, df_sea_clean]).dropna()
        
        # Clustering for hotspot logic
        coords = np.radians(df_all[['Latitude', 'Longitude']])
        db = DBSCAN(eps=500/6371.0, min_samples=20, algorithm='ball_tree', metric='haversine').fit(coords)
        df_all['Cluster'] = db.labels_
        return df_all
    except Exception as e:
        st.error(f"Mapping error: {e}")
        return pd.DataFrame(columns=['Latitude', 'Longitude'])

# Pre-load shared assets
raman_df = load_and_clean_raman()
raman_model, raman_features = train_raman_model(raman_df)
yolo_model = load_yolo_model()

# --- 3. SIDEBAR NAVIGATION ---
with st.sidebar:
    st.title("🌊 EcoTrack AI")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Dashboard Overview", 
        "🔬 Visual Inspection (CV)", 
        "🧪 Chemical Analysis (Raman)", 
        "🌍 Global Tracking (Map)"
    ])
    st.markdown("---")
    st.caption("Developed for Hackathon 2026")
    st.info("System Status: Online 🟢")

# --- 4. PAGE ROUTING ---

if page == "🏠 Dashboard Overview":
    st.title("Marine Microplastic Analysis System")
    st.write("Welcome to EcoTrack. This unified AI system automates the identification, quantification, and mapping of marine microplastics.")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Raman Samples Trained", "17,000+")
    col2.metric("Detection Precision (mAP)", "80.1%")
    col3.metric("Detection Classes", "Microplastic")
    
    st.divider()
    st.subheader("System Capabilities")
    st.markdown("""
    - **Computer Vision:** Real-time particle counting using YOLOv8.
    - **Spectroscopy:** Polymer identification (PE, PS, etc.) via Random Forest.
    - **Geospatial AI:** Density clustering to identify global pollution hotspots.
    """)

elif page == "🔬 Visual Inspection (CV)":
    st.title("Automated Particle Detection")
    st.write("Upload a microscope image. The AI will locate and count each microplastic particle.")

    uploaded_file = st.file_uploader("Upload sample image", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        
        if yolo_model:
            with st.spinner("Analyzing image pixels..."):
                results = yolo_model.predict(image_rgb, conf=0.25)
                res_plotted = results[0].plot()
                count = len(results[0].boxes)
                
                st.success(f"Analysis Complete: {count} particles detected!")
                
                c1, c2 = st.columns([3, 1])
                with c1:
                    st.image(res_plotted, caption="Detection Overlay", use_container_width=True)
                with c2:
                    st.metric("Particle Count", count)
                    if count > 20:
                        st.error("🚨 High Density Detected")
                    elif count > 5:
                        st.warning("⚠️ Moderate Presence")
                    else:
                        st.success("✅ Low Presence")
                    
                    report = f"EcoTrack Analysis\nParticles: {count}\nModel: YOLOv8-Custom"
                    st.download_button("Download Report", report, file_name="cv_report.txt")
        else:
            st.error("Model 'best.pt' not found in directory.")

elif page == "🧪 Chemical Analysis (Raman)":
    st.title("Chemical Polymer Identification")
    st.write("Identify polymer composition from Raman spectral signatures.")
    
    if st.button("Test Random Sample from Database", type="primary"):
        sample = raman_df.sample(1)
        true_label = sample['clean_category'].values[0]
        sample_features = sample[raman_features]
        prediction = raman_model.predict(sample_features)[0]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("AI Prediction")
            st.metric(label="Detected Polymer", value=prediction)
            if prediction == true_label:
                st.success("✅ Match: Verified by ground truth")
            else:
                st.error(f"❌ Mismatch. Actual: {true_label}")
        with col2:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(raman_features[:300], sample_features.values[0][:300], color='#1f77b4', lw=2)
            ax.set_title("Raman Intensity vs. Shift")
            ax.set_xlabel("Wavenumber")
            ax.set_ylabel("Intensity")
            st.pyplot(fig)

elif page == "🌍 Global Tracking (Map)":
    st.title("Global Pollution Hotspots")
    st.write("Displaying aggregated density data from global marine research.")
    
    with st.spinner("Loading geospatial clusters..."):
        map_data = generate_hotspot_map()
        if not map_data.empty:
            map_display = map_data[['Latitude', 'Longitude']].rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
            st.map(map_display, color='#ff4b4b', size=20)
            st.caption("Red markers indicate confirmed microplastic sightings across major global water bodies.")
        else:
            st.warning("No geospatial data available. Check CSV file paths.")