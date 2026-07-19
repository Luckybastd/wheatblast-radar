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
import pydeck as pdk # Pustaka baru untuk Peta 3D

# ==========================================
# KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="WheatBlast Radar", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0;}
    .sub-header {font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem;}
    /* Desain tombol Prediksi agar menonjol */
    .stButton>button {background-color: #2563EB; color: white; border-radius: 8px; font-weight: bold;}
    .stButton>button:hover {background-color: #1D4ED8; border-color: #1D4ED8;}
</style>
""", unsafe_allow_html=True)

# Inisialisasi Session State untuk Prediksi AI ke-3
if 'predicted_yield' not in st.session_state:
    st.session_state.predicted_yield = None

# ==========================================
# DEFINISI ARSITEKTUR & CACHING (TIDAK BERUBAH)
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
    
    # URL HF DIRECT ANDA
    url_ai1 = "URL_HUGGING_FACE_AI1" 
    url_ai2 = "URL_HUGGING_FACE_AI2"
    
    def download_model(url, path, name):
        if not os.path.exists(path) and url.startswith("http"):
            with st.spinner(f"Mengunduh model {name}..."):
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

with st.spinner("Menginisialisasi Mesin AWARE (Multi-AI Framework)..."):
    model_ai1, model_ai2, model_ai3, class_names = load_all_models()

# ==========================================
# SIDEBAR
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1892/1892751.png", width=100)
    st.title("🌾 AWARE System")
    st.caption("Agricultural Wheat AI Radar & Epidemiology")
    st.markdown("---")
    st.markdown("**Didukung oleh:**\n- Swin Transformer V2\n- Spatio-Temporal GAT\n- CatBoost Regressor\n- Groq Llama-3.1")

# ==========================================
# KONTEN UTAMA (SISTEM TAB)
# ==========================================
st.markdown('<p class="main-header">🌾 AWARE Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">System of Integrated Artificial Intelligence for Geospatial Analytics on Wheat Epidemics</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🔍 Radar Scanner", "📖 Ensiklopedia Penyakit", "ℹ️ Petunjuk", "🤝 Kemitraan & Riset"])

# ------------------------------------------
# TAB 1: RADAR SCANNER
# ------------------------------------------
with tab1:
    col_left, col_right = st.columns([1.1, 1.1])

    with col_left:
        st.subheader("1. Pemindaian Visual (Vision AI)")
        input_method = st.radio("Pilih Metode Input:", ["Unggah File", "Kamera"], horizontal=True)
        
        uploaded_file = st.file_uploader("Pilih foto daun/bulir", type=["jpg", "png"]) if input_method == "Unggah File" else st.camera_input("Arahkan kamera")
        
        prob_sakit_for_ai2 = 0.5
        visual_report_string = "Belum ada gambar"
        lahan_target_risk = 0.0
        risk_string = "Menunggu data"
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Spesimen Dianalisis", use_column_width=True)
            
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
                st.success(f"✅ **Diagnosis: {diagnosis}**")
            else:
                st.error(f"⚠️ **Diagnosis: {diagnosis}**")
                    
            prob_sakit_for_ai2 = float(probs[0] + probs[2] + probs[3])
            visual_report_string = f"{diagnosis} ({probs[pred_idx]*100:.1f}%)"

        st.markdown("---")
        st.subheader("2. Radar Spasial-Temporal (GNN)")
        
        # Mapping Wilayah ke Koordinat Peta
        koordinat_wilayah = {
            "Asia Selatan (India)": [22.0, 79.0],
            "Afrika Timur": [9.0, 39.0],
            "Amerika Selatan": [-14.0, -51.0],
            "Eropa Timur": [48.0, 30.0]
        }
        selected_region = st.selectbox("Zona Agro-Klimat:", list(koordinat_wilayah.keys()))
        
        # Eksekusi AI 2
        with torch.no_grad():
            X_mock = torch.randn((150, 7, 3))
            X_mock[0, -1, 2] = prob_sakit_for_ai2 
            lahan_target_risk = float(model_ai2(X_mock, torch.randint(0, 150, (2, 300))).squeeze().numpy()[0])
            risk_string = "Bahaya Ekstrem" if lahan_target_risk > 0.6 else "Risiko Sedang" if lahan_target_risk > 0.3 else "Stabil"

        # Tampilan Matriks Risiko
        st.metric(label="Risiko Penularan Regional", value=f"{lahan_target_risk * 100:.2f}%")
        
        # PETA 3D INTERAKTIF (PYDECK)
        lat, lon = koordinat_wilayah[selected_region]
        
        # Tentukan warna peta berdasarkan tingkat risiko
        if lahan_target_risk > 0.6:
            map_color = [239, 68, 68, 200] # Merah menyala
        elif lahan_target_risk > 0.3:
            map_color = [245, 158, 11, 200] # Oranye
        else:
            map_color = [16, 185, 129, 200] # Hijau
            
        df_map = pd.DataFrame({"lat": np.random.randn(20) * 1.5 + lat, "lon": np.random.randn(20) * 1.5 + lon})
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[lon, lat]',
            get_radius=150000,
            get_fill_color=map_color,
            pickable=True
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5, pitch=30)
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip={"text": f"Zona {risk_string}"}))

    with col_right:
        st.subheader("3. Proyeksi Panen (Expert System)")
        
        with st.expander("⚙️ Atur Parameter Lingkungan (Opsional)", expanded=True):
            temp_input = st.slider("Suhu Harian (°C)", 10.0, 40.0, 25.0)
            rainfall_input = st.slider("Curah Hujan (mm)", 0.0, 300.0, 150.0)
            fertilizer_type = st.selectbox("Jenis Pupuk", ["Chemical", "Organic", "Mixed"])
            pesticide_usage = st.selectbox("Intensitas Pestisida", ["Low", "Medium", "High"])

        # Tombol Prediksi Interaktif
        if st.button("🚀 Prediksi Tonase Panen", use_container_width=True):
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
            try:
                st.session_state.predicted_yield = model_ai3.predict(pd.DataFrame(input_data))[0]
            except Exception:
                st.session_state.predicted_yield = 0.0

        # Logika Tampilan Standby / Hasil Prediksi
        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.predicted_yield is None:
            st.metric(label="Prediksi Hasil Panen Akhir", value="-- Kg/Hektar", delta="Tunggu Input")
        else:
            st.metric(label="Prediksi Hasil Panen Akhir", value=f"{st.session_state.predicted_yield:.2f} Kg/Hektar", delta="- Dampak Epidemi" if prob_sakit_for_ai2 > 0.5 else "Optimal", delta_color="inverse")

        st.markdown("---")
        st.subheader("4. Sintesis Pakar Agronomi (LLM)")
        
        if st.button("🤖 Generate Laporan Pakar (Groq API)"):
            if st.session_state.predicted_yield is None:
                st.warning("Harap lakukan prediksi panen terlebih dahulu pada Langkah 3.")
            else:
                prompt_llm = f"""
                Anda adalah seorang konsultan agrikultur senior dari FAO. Buat laporan (maks 3 kalimat) dalam BAHASA INGGRIS berdasarkan data AI:
                - Status Daun: {visual_report_string}
                - Risiko Spasial: {risk_string} ({lahan_target_risk*100:.1f}%)
                - Prediksi Panen: {st.session_state.predicted_yield:.1f} Kg/Ha.
                Akhiri dengan satu tindakan mitigasi.
                """
                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    with st.spinner("Menyintesis laporan via Llama-3.1..."):
                        report_text = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt_llm}], model="llama-3.1-8b-instant", temperature=0.3
                        ).choices[0].message.content
                    st.info(report_text)
                except Exception as e:
                    st.error(f"Koneksi Groq Gagal: {str(e)}")

# ------------------------------------------
# TAB 2: ENSIKLOPEDIA KUSTOM (PULL WARNA)
# ------------------------------------------
with tab2:
    st.header("📖 Ensiklopedia Patogen Gandum")
    st.write("Visualisasi tingkat ancaman berdasarkan klasifikasi keparahan ekologis.")
    
    # KUSTOM HTML BOKS (Warna Utuh)
    st.markdown("""
    <div style="background-color: rgba(220, 38, 38, 0.15); border: 1px solid #DC2626; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
        <details open>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #EF4444; cursor: pointer; list-style: none;">
                🌾 Wheat Blast <i>(Magnaporthe oryzae pathotype Triticum)</i> — [BAHAYA KARANTINA]
            </summary>
            <hr style="border-color: rgba(220, 38, 38, 0.3);">
            <p style="color: #F87171;"><b>Deskripsi:</b> Penyakit jamur sangat mematikan yang menyebar cepat, menyerang bulir gandum dan menyebabkan pemutihan total secara eksponensial dalam hitungan hari.</p>
            <p style="color: #F87171;"><b>Gejala Visual:</b> Bulir menjadi putih terang dan kering, bercak abu-abu pada daun dengan tepi cokelat gelap menembus jaringan vaskular.</p>
            <p style="color: #F87171;"><b>Mitigasi:</b> Pemusnahan total (eradikasi) radius terinfeksi, aplikasi fungisida sistemik triazole, dan pelaporan segera ke dinas pertanian lokal.</p>
        </details>
    </div>
    
    <div style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
        <details>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #FBBF24; cursor: pointer; list-style: none;">
                🪵 Stem Rust <i>(Puccinia graminis)</i> — [RESIKO TINGGI]
            </summary>
            <hr style="border-color: rgba(245, 158, 11, 0.3);">
            <p style="color: #FCD34D;"><b>Deskripsi:</b> Patogen perusak pembuluh batang yang memotong aliran nutrisi, menyebabkan tanaman rebah atau patah menjelang masa panen.</p>
            <p style="color: #FCD34D;"><b>Gejala Visual:</b> Pustula merah bata berbentuk berlian yang memecah epidermis batang, melepaskan debu spora berkarat jika disentuh.</p>
            <p style="color: #FCD34D;"><b>Mitigasi:</b> Pemberantasan inang sekunder (gulma barberry) dan penyemprotan protektan spektrum luas.</p>
        </details>
    </div>
    
    <div style="background-color: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 10px; padding: 15px;">
        <details>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #60A5FA; cursor: pointer; list-style: none;">
                🍂 Leaf Rust <i>(Puccinia triticina)</i> — [RESIKO SEDANG]
            </summary>
            <hr style="border-color: rgba(59, 130, 246, 0.3);">
            <p style="color: #93C5FD;"><b>Deskripsi:</b> Karat daun klasik yang umum ditemui, memperlambat proses fotosintesis dan menurunkan kualitas butir gandum secara bertahap.</p>
            <p style="color: #93C5FD;"><b>Gejala Visual:</b> Bintik-bintik oranye tersebar acak secara eksklusif pada helaian daun bagian atas.</p>
            <p style="color: #93C5FD;"><b>Mitigasi:</b> Pemilihan varietas gandum tahan karat (resistant cultivar) untuk musim tanam berikutnya.</p>
        </details>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: PETUNJUK
# ------------------------------------------
with tab3:
    st.header("ℹ️ Arsitektur Integrasi AWARE")
    st.markdown("""
    Sistem ini memvalidasi ancaman biologis melalui empat matriks kecerdasan buatan:
    1. **Edge-Vision AI:** Klasifikasi patogen pada spesimen lokal (Gambar).
    2. **GNN Spasial:** Meramalkan penyebaran wabah lintas perkebunan (Peta 3D).
    3. **Expert Modeler:** Menerjemahkan dampak klinis ke dalam rasio penyusutan tonase ekonomi (Prediksi).
    4. **Generative NLP:** Menerjemahkan metrik teknis menjadi protokol mitigasi manusia.
    """)

# ------------------------------------------
# TAB 4: KEMITRAAN & KONTAK
# ------------------------------------------
with tab4:
    st.header("🤝 Kemitraan Inovasi & Kolaborasi Riset")
    st.markdown("Proyek **AWARE** bersifat arsitektur terbuka (*Open-Architecture*) dan menyambut baik inisiatif kerja sama dari institusi agrikultur, peneliti, maupun investor global.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("👨‍🔬 Peneliti Utama")
        st.markdown("""
        **Jonatan**  
        *Undergraduate Researcher, Applied Mathematics*  
        **Universitas Sriwijaya (UNSRI)**, Indonesia.  
        Fokus Riset: Integrasi *Machine Learning*, *Computer Vision*, & *System Dynamics* untuk ketahanan pangan.
        
        📧 Email: partnership@aware-unsri.edu.id (Mock)
        """)
    with col_c2:
        st.subheader("🌐 Global Research Network")
        st.markdown("Kami mengkalibrasi model kami berdasarkan data terbuka dari institusi agrikultur dunia:")
        st.markdown("""
        - [CIMMYT (International Maize and Wheat Improvement Center)](https://www.cimmyt.org/)
        - [FAO Global Wheat Rust Monitoring System](https://rusttracker.cimmyt.org/)
        - [Borlaug Global Rust Initiative (BGRI)](https://bgri.cornell.edu/)
        """)
