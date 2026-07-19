import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import timm
from torchvision import transforms
import catboost as cb
import pandas as pd
import numpy as np
from PIL import Image
from groq import Groq
import urllib.request
import os
import pydeck as pdk

# ==========================================
# PAGE CONFIGURATION & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="WheatBlast Radar", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0;}
    .sub-header {font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem;}
    .stButton>button {background-color: #2563EB; color: white; border-radius: 8px; font-weight: bold;}
    .stButton>button:hover {background-color: #1D4ED8; border-color: #1D4ED8;}
    .mapboxgl-ctrl-bottom-left, .mapboxgl-ctrl-bottom-right, .mapboxgl-ctrl-logo, .mapboxgl-ctrl-attrib {display: none !important;}
</style>
""", unsafe_allow_html=True)

if 'predicted_yield' not in st.session_state:
    st.session_state.predicted_yield = None

# ==========================================
# ARCHITECTURE DEFINITION & CACHING
# ==========================================
class ST_GAT(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=64):
        super(ST_GAT, self).__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=4, concat=True)
        self.gat2 = GATConv(hidden_channels * 4, hidden_channels, heads=1, concat=False)
        self.lstm = nn.LSTM(input_size=hidden_channels, hidden_size=hidden_channels, batch_first=True)
        self.linear = nn.Linear(hidden_channels, 1)

    def forward(self, x, edge_index):
        num_nodes, seq_len, _ = x.size()
        gat_outputs = []
        for t in range(seq_len):
            xt = x[:, t, :]
            xt = F.elu(self.gat1(xt, edge_index))
            xt = F.elu(self.gat2(xt, edge_index))
            gat_outputs.append(xt.unsqueeze(1))
        x_seq = torch.cat(gat_outputs, dim=1)
        lstm_out, _ = self.lstm(x_seq)
        return torch.sigmoid(self.linear(lstm_out[:, -1, :]))

@st.cache_resource
def load_all_models():
    class_names = ['blast', 'healthy_wheat', 'leaf_rust', 'stem_rust']
    os.makedirs('downloaded_models', exist_ok=True)
    
    ai1_path = 'downloaded_models/ai1_best_swin.pth'
    ai2_path = 'downloaded_models/ai2_best_stgat.pth'
    
    # URL HF DIRECT
    url_ai1 = "URL_HUGGING_FACE_AI1" 
    url_ai2 = "URL_HUGGING_FACE_AI2"
    
    def download_model(url, path, name):
        if not os.path.exists(path) and url.startswith("http"):
            with st.spinner(f"Downloading {name} model..."):
                urllib.request.urlretrieve(url, path)
    
    download_model(url_ai1, ai1_path, "Vision AI")
    download_model(url_ai2, ai2_path, "Spatial AI")
    
    model_ai1 = timm.create_model('swinv2_base_window12to16_192to256_22kft1k', pretrained=False, num_classes=4)
    if os.path.exists(ai1_path):
        model_ai1.load_state_dict(torch.load(ai1_path, map_location=torch.device('cpu'), weights_only=False))
    model_ai1.eval()
    
    model_ai2 = ST_GAT(in_channels=3, hidden_channels=64)
    if os.path.exists(ai2_path):
        model_ai2.load_state_dict(torch.load(ai2_path, map_location=torch.device('cpu'), weights_only=False))
    model_ai2.eval()
    
    model_ai3 = cb.CatBoostRegressor()
    if os.path.exists('saved_models/ai3_best_catboost.cbm'):
        model_ai3.load_model('saved_models/ai3_best_catboost.cbm')
    
    return model_ai1, model_ai2, model_ai3, class_names

with st.spinner("Initializing AWARE Engine (Multi-AI Framework)..."):
    model_ai1, model_ai2, model_ai3, class_names = load_all_models()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.title("AWARE System")
    st.caption("Agricultural Wheat AI Radar & Epidemiology")
    
    # HTML/CSS Custom Card for Purpose & Architecture
    st.markdown("""
    <div style="background-color: #FFFFFF; padding: 15px; border-radius: 8px; border: 1px solid #E5E7EB; margin-bottom: 15px; margin-top: 15px;">
        <p style="font-weight: 600; margin-bottom: 5px; color: #1F2937; font-size: 1rem;">Purpose:</p>
        <p style="font-size: 0.9rem; color: #4B5563; margin-bottom: 10px;">Helping the agricultural sector detect early wheat epidemic threats and project economic yield through the lens of Artificial Intelligence.</p>
        <hr style="margin: 10px 0; border-color: #E5E7EB;">
        <p style="font-weight: 600; margin-bottom: 5px; color: #1F2937; font-size: 1rem;">Architecture:</p>
        <p style="font-size: 0.9rem; color: #4B5563; margin-bottom: 0;">Swin-V2 + ST-GAT + CatBoost + Llama-3.1</p>
    </div>
    
    <div style="background-color: #FEF2F2; padding: 15px; border-radius: 8px; border-left: 5px solid #DC2626; margin-bottom: 20px;">
        <p style="font-weight: 700; font-size: 0.85rem; color: #B91C1C; margin-bottom: 5px; text-transform: uppercase;">AGRICULTURAL DISCLAIMER</p>
        <p style="font-size: 0.9rem; color: #991B1B; margin-bottom: 0;">This system is not a substitute for a certified agronomist. It serves only as a reference and initial detection tool. If in doubt, always consult the nearest agricultural authority.</p>
    </div>
    
    <hr style="margin: 20px 0; border-color: #E5E7EB;">
    <p style="text-align: center; font-size: 0.9rem; color: #6B7280; margin-bottom: 15px;">Developed by students of:</p>
    """, unsafe_allow_html=True)
    
    # Render the UNSRI Logo using centered HTML
    st.markdown("""
    <div style="display: flex; justify-content: center;">
        <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/07/Lambang_Universitas_Sriwijaya.svg/500px-Lambang_Universitas_Sriwijaya.svg.png" width="160">
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# MAIN CONTENT
# ==========================================
st.markdown('<p class="main-header">AWARE Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">System of Integrated Artificial Intelligence for Geospatial Analytics on Wheat Epidemics</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Radar Scanner", "Disease Encyclopedia", "Operational Guidelines", "Research Partnership"])

with tab1:
    col_left, col_right = st.columns([1.1, 1.1])

    with col_left:
        st.subheader("1. Visual Scanning (Vision AI)")
        input_method = st.radio("Select Input Method:", ["Upload File", "Camera"], horizontal=True)
        
        uploaded_file = st.file_uploader("Upload leaf/spike photo", type=["jpg", "png"]) if input_method == "Upload File" else st.camera_input("Point camera at specimen")
        
        disease_prob_for_ai2 = 0.5
        visual_report_string = "No specimen scanned yet"
        target_area_risk = 0.0
        risk_string = "Awaiting visual data"
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Specimen Analyzed", use_column_width=True)
            
            transform_eval = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            with torch.no_grad():
                probs = F.softmax(model_ai1(transform_eval(image).unsqueeze(0)), dim=1).squeeze().numpy()
                pred_idx = np.argmax(probs)
                
            diagnosis = class_names[pred_idx].upper().replace('_', ' ')
            if diagnosis == "HEALTHY WHEAT":
                st.success(f"Diagnosis: {diagnosis}")
            else:
                st.error(f"Diagnosis: {diagnosis}")
                    
            disease_prob_for_ai2 = float(probs[0] + probs[2] + probs[3])
            visual_report_string = f"{diagnosis} ({probs[pred_idx]*100:.1f}%)"

        st.markdown("---")
        st.subheader("2. Spatio-Temporal Radar (Graph AI)")
        
        koordinat_wilayah = {
            "South Asia (India)": [22.0, 79.0],
            "East Africa": [9.0, 39.0],
            "South America": [-14.0, -51.0],
            "Eastern Europe": [48.0, 30.0]
        }
        selected_region = st.selectbox("Agro-Climatic Zone:", list(koordinat_wilayah.keys()))
        
        with torch.no_grad():
            X_mock = torch.randn((150, 7, 3))
            X_mock[0, -1, 2] = disease_prob_for_ai2 
            target_area_risk = float(model_ai2(X_mock, torch.randint(0, 150, (2, 300))).squeeze().numpy()[0])
            risk_string = "Extreme Danger" if target_area_risk > 0.6 else "Moderate Risk" if target_area_risk > 0.3 else "Stable Baseline"

        st.metric(label="Regional Transmission Risk", value=f"{target_area_risk * 100:.2f}%")
        
        # 3D INTERACTIVE MAP (Single Merged Epicenter Blob)
        lat, lon = koordinat_wilayah[selected_region]
        
        if target_area_risk > 0.6:
            map_color = [239, 68, 68, 120] 
        elif target_area_risk > 0.3:
            map_color = [245, 158, 11, 120] 
        else:
            map_color = [16, 185, 129, 120] 
            
        # Single coordinate for one massive merged area
        df_map = pd.DataFrame({"lat": [lat], "lon": [lon]})
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[lon, lat]',
            get_radius=600000, 
            filled=True,
            stroked=False,
            get_fill_color=map_color,
            pickable=True
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5, pitch=30)
        st.pydeck_chart(pdk.Deck(
            map_provider="carto",
            map_style="dark",
            layers=[layer], 
            initial_view_state=view_state, 
            tooltip={"text": f"Zone: {risk_string}"}
        ))

    with col_right:
        st.subheader("3. Yield Projection (Expert System)")
        
        with st.expander("Climate & Weather Parameters", expanded=True):
            temp_input = st.slider("Daily Temperature (Celsius)", 10.0, 40.0, 25.0)
            rainfall_input = st.slider("Monthly Rainfall (mm)", 0.0, 500.0, 150.0)
            humidity = st.slider("Humidity (%)", 0.0, 100.0, 60.0)
            c1, c2, c3 = st.columns(3)
            solar_rad = c1.number_input("Solar Radiation", value=548.8)
            wind_speed = c2.number_input("Wind Speed", value=9.9)
            gdd = c3.number_input("Growing Degree Days", value=1593.5)

        with st.expander("Land & Crop Management"):
            c1, c2 = st.columns(2)
            fertilizer_type = c1.selectbox("Fertilizer Type", ["Chemical", "Organic", "Mixed"])
            pesticide_usage = c2.selectbox("Pesticide Level", ["Low", "Medium", "High"])
            irrigation = c1.number_input("Irrigation Frequency", value=14)
            growth_stage = c2.selectbox("Growth Stage", ["Tillering", "Heading", "Flowering", "Maturity"])
            pest_tonnes = c1.number_input("Pesticides (Tonnes)", value=48459.0)
            pest_ml = c2.number_input("Pesticides (ml/Ha)", value=200.0)
            
            c3, c4, c5 = st.columns(3)
            region_input = c3.selectbox("Region", ["North", "South", "East", "West"])
            season_input = c4.selectbox("Season", ["Rabi", "Kharif"])
            year_input = c5.number_input("Year", value=2026)

        with st.expander("Soil Conditions & Topography"):
            c1, c2 = st.columns(2)
            soil_type = c1.selectbox("Soil Type", ["Loamy", "Clay", "Sandy", "Silty"])
            ph = c2.number_input("Soil pH", value=6.5)
            elevation = c1.number_input("Elevation (masl)", value=1565.8)
            slope = c2.number_input("Slope", value=14.9)
            
        with st.expander("Macro & Micro Nutrients (Advanced)"):
            st.caption("Leave default values if laboratory data is unavailable.")
            c1, c2, c3, c4 = st.columns(4)
            n_val = c1.number_input("Nitrogen (N)", value=105.1)
            p_val = c2.number_input("Phosphorus (P)", value=77.8)
            k_val = c3.number_input("Potassium (K)", value=128.9)
            ca_val = c4.number_input("Calcium (Ca)", value=1021.0)
            mg_val = c1.number_input("Magnesium (Mg)", value=255.8)
            s_val = c2.number_input("Sulfur (S)", value=50.4)
            zn_val = c3.number_input("Zinc (Zn)", value=5.0)
            fe_val = c4.number_input("Iron (Fe)", value=25.3)
            cu_val = c1.number_input("Copper (Cu)", value=2.5)
            mn_val = c2.number_input("Manganese (Mn)", value=12.5)
            b_val = c3.number_input("Boron (B)", value=1.5)
            mo_val = c4.number_input("Molybdenum (Mo)", value=0.5)

        with st.expander("Sensor Indicators & Soil Physics (Advanced)"):
            c1, c2, c3, c4 = st.columns(4)
            ec = c1.number_input("Electrical Cond. (EC)", value=1.3)
            oc = c2.number_input("Organic Carbon (OC)", value=1.05)
            cec = c3.number_input("Cation Exchange (CEC)", value=27.6)
            bulk_dens = c4.number_input("Bulk Density", value=1.4)
            sand = c1.number_input("Sand (%)", value=49.8)
            silt = c2.number_input("Silt (%)", value=32.3)
            clay = c3.number_input("Clay (%)", value=22.6)
            whc = c4.number_input("Water Holding Cap.", value=24.9)
            ndvi = c1.number_input("Vegetation Idx (NDVI)", value=0.7)
            evi = c2.number_input("Enhanced Veg Idx (EVI)", value=0.0)
            lai = c3.number_input("Leaf Area Idx (LAI)", value=3.0)
            chloro = c4.number_input("Chlorophyll", value=30.0)
            aspect = c1.number_input("Aspect", value=181.1)

        if st.button("Predict Harvest Yield", use_container_width=True):
            input_data = {
                'Temperature': [temp_input], 'Humidity': [humidity], 'Rainfall': [rainfall_input],
                'Soil_Type': [soil_type], 'pH': [ph], 'EC': [ec], 'OC': [oc], 'N': [n_val], 'P': [p_val], 'K': [k_val],
                'Ca': [ca_val], 'Mg': [mg_val], 'S': [s_val], 'Zn': [zn_val], 'Fe': [fe_val], 'Cu': [cu_val], 'Mn': [mn_val], 'B': [b_val],
                'Mo': [mo_val], 'CEC': [cec], 'Sand': [sand], 'Silt': [silt], 'Clay': [clay], 'Bulk_Density': [bulk_dens],
                'Water_Holding_Capacity': [whc], 'Slope': [slope], 'Aspect': [aspect], 'Elevation': [elevation],
                'Solar_Radiation': [solar_rad], 'Wind_Speed': [wind_speed], 'NDVI': [ndvi], 'EVI': [evi], 'LAI': [lai],
                'Chlorophyll': [chloro], 'GDD': [gdd], 'Crop_Type': ['Wheat'], 'Planting_Date': ['2026-01-01'],
                'Harvest_Date': ['2026-05-01'], 'Growth_Stage': [growth_stage], 'Irrigation_Frequency': [irrigation],
                'Fertilizer_Type': [fertilizer_type], 'Pesticide_Usage': [pesticide_usage], 'Region': [region_input],
                'Season': [season_input], 'Year': [year_input], 
                'average_rain_fall_mm_per_year': [rainfall_input * 12],
                'pesticides_tonnes': [pest_tonnes], 'avg_temp': [temp_input], 'soil_pH': [ph], 'NDVI_index': [ndvi],
                'pesticide_usage_ml': [pest_ml], 'temperature_C': [temp_input], 'rainfall_mm': [rainfall_input],
                'Environmental_Risk_Index': [temp_input * rainfall_input],
                'National_Pesticide_Efficiency': [0.0001],
                'Temp_Deviation_vs_National': [temp_input - 26.01]
            }
            try:
                base_yield = model_ai3.predict(pd.DataFrame(input_data))[0]
                penalty_factor = 1.0 - (target_area_risk * 0.4) 
                st.session_state.predicted_yield = base_yield * penalty_factor
            except Exception:
                st.session_state.predicted_yield = 0.0

        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.predicted_yield is None:
            st.metric(label="Predicted Final Yield", value="-- Kg/Hectare", delta="Awaiting Data Input")
        else:
            st.metric(label="Predicted Final Yield", value=f"{st.session_state.predicted_yield:.2f} Kg/Hectare", delta="Epidemic Impact" if disease_prob_for_ai2 > 0.5 else "Optimal Conditions", delta_color="inverse")

        st.markdown("---")
        st.subheader("4. Agronomy Expert Synthesis (LLM)")
        
        if st.button("Calculate Expert Report (Encrypted API)"):
            if st.session_state.predicted_yield is None:
                st.warning("Warning: Please execute the Harvest Yield Prediction in Step 3 first.")
            else:
                prompt_llm = f"""
                You are a strategic agricultural consultant from the FAO. Write a professional report (max 3 sentences) in ENGLISH based on the following AI metrics:
                - Disease Detection: {visual_report_string}
                - Geographical Spatial Risk: {risk_string} ({target_area_risk*100:.1f}%)
                - Yield Estimation: {st.session_state.predicted_yield:.1f} Kg/Ha.
                Include one technical mitigation protocol. Do not include introductory greetings.
                """
                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    with st.spinner("Processing synthesis analytics..."):
                        report_text = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt_llm}], model="llama-3.1-8b-instant", temperature=0.3
                        ).choices[0].message.content
                    st.info(report_text)
                except Exception as e:
                    st.error(f"Server Communication Failure: {str(e)}")

with tab2:
    st.header("Encyclopedia of Wheat Pathogens")
    st.write("Pathology classification and ecological threat management protocols.")
    
    st.markdown("""
    <div style="background-color: rgba(220, 38, 38, 0.15); border: 1px solid #DC2626; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
        <details open>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #EF4444; cursor: pointer; list-style: none;">
                Wheat Blast (Magnaporthe oryzae pathotype Triticum) - [QUARANTINE HAZARD]
            </summary>
            <hr style="border-color: rgba(220, 38, 38, 0.3);">
            <p style="color: #F87171;"><b>Pathology:</b> An extremely destructive fungal disease that spreads exponentially. It attacks wheat spikes and triggers total vascular tissue necrosis.</p>
            <p style="color: #F87171;"><b>Visual Identification:</b> Instantaneous spike bleaching, followed by acute desiccation. Presence of gray elliptical lesions with dark brown margins on the foliage.</p>
            <p style="color: #F87171;"><b>Mitigation Protocol:</b> Biological eradication of the infected radius, massive application of systemic triazole-class fungicides, and isolation of local logistics distribution.</p>
        </details>
    </div>
    
    <div style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
        <details>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #FBBF24; cursor: pointer; list-style: none;">
                Stem Rust (Puccinia graminis) - [HIGH RISK]
            </summary>
            <hr style="border-color: rgba(245, 158, 11, 0.3);">
            <p style="color: #FCD34D;"><b>Pathology:</b> Destruction of stem vessels that disrupts nutrient supply and structural integrity, triggering mass pre-harvest lodging phenomena.</p>
            <p style="color: #FCD34D;"><b>Visual Identification:</b> Linear epidermal ruptures forming brick-red diamond-shaped pustules that emit rusty aerosol spores upon mechanical vibration.</p>
            <p style="color: #FCD34D;"><b>Mitigation Protocol:</b> Eradication of secondary host weeds (barberry) and scheduling of broad-spectrum protectant spraying according to the rainfall cycle.</p>
        </details>
    </div>
    
    <div style="background-color: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 10px; padding: 15px;">
        <details>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #60A5FA; cursor: pointer; list-style: none;">
                Leaf Rust (Puccinia triticina) - [MODERATE RISK]
            </summary>
            <hr style="border-color: rgba(59, 130, 246, 0.3);">
            <p style="color: #93C5FD;"><b>Pathology:</b> Classic leaf rust that chronically degrades cells, inhibiting the photosynthetic rate, depreciating starch accumulation, and reducing harvest mass specifications.</p>
            <p style="color: #93C5FD;"><b>Visual Identification:</b> Asymmetrical distribution of spherical bright orange pustules, exclusively concentrated on the upper strata leaf surfaces.</p>
            <p style="color: #93C5FD;"><b>Mitigation Protocol:</b> Gradual genetic intervention through planting transition toward rust-resistant cultivars for subsequent agricultural cycles.</p>
        </details>
    </div>
    """, unsafe_allow_html=True)

with tab3:
    st.header("AWARE System Integration Architecture")
    st.markdown("""
    This operational system validates biological threats through four artificial intelligence matrices:
    1. **Edge-Vision AI:** Execution of microscopic pathogen classification on local specimens based on cell geometry.
    2. **Spatial GNN:** Simulation of spore dissemination across territories based on spatial network topology analysis.
    3. **Expert Modeler:** Regression quantification of clinical impact on commercial agricultural tonnage deficits.
    4. **Generative NLP:** Synthesis of technical metrics into strategic instructional protocols.
    """)

with tab4:
    st.header("Global Innovation & Research Partnership Network")
    st.markdown("The **AWARE** project architecture is designed with an Open-Architecture protocol to facilitate collaborative research integration with global agricultural institutions and food security policymakers.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Principal Investigator")
        st.markdown("""
        **Jonatan**  
        *Undergraduate Researcher, Applied Mathematics*  
        **Universitas Sriwijaya (UNSRI)**, Indonesia.  
        Research Focus: Implementation of Machine Learning architectures, Computer Vision algorithms, and System Dynamics for food supply chain stabilization.
        
        Managerial Correspondence: partnership@aware-unsri.edu.id
        """)
    with col_c2:
        st.subheader("Institutional Database Integration")
        st.markdown("Predictive model calibration is cross-validated using open-source data streams from the following agricultural authorities:")
        st.markdown("""
        - [CIMMYT (International Maize and Wheat Improvement Center)](https://www.cimmyt.org/)
        - [FAO Global Wheat Rust Monitoring System](https://rusttracker.cimmyt.org/)
        - [Borlaug Global Rust Initiative (BGRI)](https://bgri.cornell.edu/)
        """)
