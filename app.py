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
from groq import Groq # Pustaka baru untuk Cloud Llama-3

st.set_page_config(page_title="WheatBlast Radar", page_icon="🌾", layout="wide")

# ==========================================
# DEFINISI ARSITEKTUR MODEL
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

# ==========================================
# LOAD MODELS SECARA AMAN (CPU ONLY UNTUK CLOUD)
# ==========================================
@st.cache_resource
def load_all_models():
    class_names = ['blast', 'healthy_wheat', 'leaf_rust', 'stem_rust']
    
    # 1. Load AI 1 (Swin) - Paksa ke CPU
    model_ai1 = timm.create_model('swinv2_base_window12to16_192to256_22kft1k', pretrained=False, num_classes=4)
    model_ai1.load_state_dict(torch.load('saved_models/ai1_best_swin.pth', map_location=torch.device('cpu')))
    model_ai1.eval()
    
    # 2. Load AI 2 (ST-GAT) - Paksa ke CPU
    model_ai2 = ST_GAT(in_channels=3, hidden_channels=64)
    model_ai2.load_state_dict(torch.load('saved_models/ai2_best_stgat.pth', map_location=torch.device('cpu'), weights_only=False))
    model_ai2.eval()
    
    # 3. Load AI 3 (CatBoost)
    model_ai3 = cb.CatBoostRegressor()
    model_ai3.load_model('saved_models/ai3_best_catboost.cbm')
    
    return model_ai1, model_ai2, model_ai3, class_names

with st.spinner("Initializing Enterprise Multi-AI Pipeline on Cloud..."):
    model_ai1, model_ai2, model_ai3, class_names = load_all_models()

st.title("🌾 Multi-AI Framework for Wheat Epidemic Radar")
st.markdown("---")

col_left, col_right = st.columns([1, 1])

with col_left:
    st.header("🔬 Step 1 & 2: Computer Vision & Spatial")
    uploaded_file = st.file_uploader("Upload Wheat Image:", type=["jpg", "jpeg", "png"])
    
    prob_sakit_for_ai2 = 0.5
    visual_report_string = "No Image Uploaded"
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Specimen", width=350)
        
        transform_eval = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        input_tensor = transform_eval(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = model_ai1(input_tensor)
            probs = F.softmax(outputs, dim=1).squeeze().numpy()
            pred_idx = np.argmax(probs)
            
        st.success(f"**Diagnostic: {class_names[pred_idx].upper()}**")
        prob_sakit_for_ai2 = float(probs[0] + probs[2] + probs[3])
        visual_report_string = f"{class_names[pred_idx].upper()} ({probs[pred_idx]*100:.1f}%)"

    st.markdown("---")
    st.subheader("🌐 Spatio-Temporal Risk")
    selected_region = st.selectbox("Agro-Climatic Zone:", ["North India", "South India", "East Africa", "South USA"])
    
    num_nodes = 150
    edge_index_mock = torch.randint(0, num_nodes, (2, 300))
    X_mock = torch.randn((num_nodes, 7, 3))
    X_mock[0, -1, 2] = prob_sakit_for_ai2 
    
    with torch.no_grad():
        graph_risk_output = model_ai2(X_mock, edge_index_mock).squeeze().numpy()
        lahan_target_risk = float(graph_risk_output[0])
        
    st.metric(label=f"Outbreak Risk in {selected_region}", value=f"{lahan_target_risk * 100:.2f}%")
    risk_string = "High Danger" if lahan_target_risk > 0.6 else "Stable Baseline"

with col_right:
    st.header("📊 Step 3 & 4: Yield Expert & LLM Brief")
    
    c1, c2 = st.columns(2)
    with c1:
        temp_input = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
        rainfall_input = st.slider("Rainfall (mm)", 0.0, 300.0, 150.0)
    with c2:
        fertilizer_type = st.selectbox("Fertilizer", ["Chemical", "Organic", "Mixed"])
        pesticide_usage = st.selectbox("Pesticide", ["Low", "Medium", "High"])

    input_data = {
        'Temperature': [temp_input], 'Humidity': [60.0], 'Rainfall': [rainfall_input],
        'Soil_Type': ['Loamy'], 'pH': [6.5], 'EC': [1.3], 'OC': [1.05], 'N': [105.1], 'P': [77.8], 'K': [128.9],
        'Ca': [1021.0], 'Mg': [255.8], 'S': [50.4], 'Zn': [5.0], 'Fe': [25.3], 'Cu': [2.5], 'Mn': [12.5], 'B': [1.5],
        'Mo': [0.5], 'CEC': [27.6], 'Sand': [49.8], 'Silt': [32.3], 'Clay': [22.6], 'Bulk_Density': [1.4],
        'Water_Holding_Capacity': [24.9], 'Slope': [14.9], 'Aspect': [181.1], 'Elevation': [1565.8],
        'Solar_Radiation': [548.8], 'Wind_Speed': [9.9], 'NDVI': [0.0], 'EVI': [0.0], 'LAI': [3.0],
        'Chlorophyll': [30.0], 'GDD': [1593.5], 'Crop_Type': ['Wheat'], 'Planting_Date': ['2026-01-01'],
        'Harvest_Date': ['2026-05-01'], 'Growth_Stage': ['Maturity'], 'Irrigation_Frequency': [14],
        'Fertilizer_Type': [fertilizer_type], 'Pesticide_Usage': [pesticide_usage], 'Region': ['North'],
        'Season': ['Rabi'], 'Year': [2026], 'average_rain_fall_mm_per_year': [1083.0],
        'pesticides_tonnes': [48459.04], 'avg_temp': [26.01], 'soil_pH': [6.5], 'NDVI_index': [0.7],
        'pesticide_usage_ml': [200.0], 'temperature_C': [temp_input], 'rainfall_mm': [rainfall_input],
        'Environmental_Risk_Index': [temp_input * rainfall_input],
        'National_Pesticide_Efficiency': [0.0001],
        'Temp_Deviation_vs_National': [temp_input - 26.01]
    }
    
    predicted_yield = model_ai3.predict(pd.DataFrame(input_data))[0]
    st.metric(label="Predicted Yield Quantity", value=f"{predicted_yield:.2f} Kg/Ha")

    st.markdown("---")
    st.subheader("🤖 Autonomous Senior Agronomist Brief")
    
    # KUNCI DEPLOYMENT WEB: MENGGUNAKAN CLOUD API (GROQ) BUKAN LOCALHOST
    if st.button("Synthesize Expert Report"):
        prompt_llm = f"""
        You are an elite agricultural consultant. Write a precise, highly authoritative agronomy brief summary (maximum 3 sentences) in English based on the following AI analytics:
        - Image Diagnostic (Swin-V2): {visual_report_string}
        - Transmission Risk (ST-GAT Graph): {risk_string} ({lahan_target_risk*100:.1f}%)
        - Predicted Yield Output (CatBoost): {predicted_yield:.1f} Kg/Ha
        End with one mitigation action.
        """
        
        try:
            # Pengecekan spesifik untuk memastikan Secret terbaca
            api_key = st.secrets["GROQ_API_KEY"]
            client = Groq(api_key=api_key)
            
            with st.spinner("Connecting to Llama-3 Cloud Brain..."):
                chat_completion = client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt_llm}],
                    model="llama-3.1-8b-instant", 
                    temperature=0.3
                )
            report_text = chat_completion.choices[0].message.content
            st.info(report_text)
            
        except KeyError:
            # Jika benar-benar Streamlit belum membaca API Key
            st.error("Kunci API belum terbaca oleh sistem. Streamlit membutuhkan waktu sekitar 1-2 menit untuk memproses Secrets baru. Silakan Refresh (F5) halaman web Anda.")
            
        except Exception as e:
            # Menampilkan ERROR ASLI dari server Groq
            st.error(f"Kunci API terbaca, tetapi terjadi kesalahan koneksi/model: {str(e)}")
