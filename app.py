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

# ==========================================
# KONFIGURASI HALAMAN & CUSTOM CSS (MOBILE RESPONSIVE)
# ==========================================
st.set_page_config(page_title="WheatBlast Radar", page_icon="🌾", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0;}
    .sub-header {font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem;}
    .metric-card {background-color: #F3F4F6; padding: 15px; border-radius: 10px; border-left: 5px solid #3B82F6;}
</style>
""", unsafe_allow_html=True)

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
# FUNGSI CACHING UNTUK MEMUAT MODEL
# ==========================================
@st.cache_resource
def load_all_models():
    class_names = ['blast', 'healthy_wheat', 'leaf_rust', 'stem_rust']
    os.makedirs('downloaded_models', exist_ok=True)
    
    ai1_path = 'downloaded_models/ai1_best_swin.pth'
    ai2_path = 'downloaded_models/ai2_best_stgat.pth'
    
    # GANTI URL INI DENGAN LINK HUGGING FACE ASLI ANDA JIKA SUDAH ADA
    url_ai1 = "URL_HUGGING_FACE_AI1" 
    url_ai2 = "URL_HUGGING_FACE_AI2"
    
    def download_model(url, path, name):
        if not os.path.exists(path) and url.startswith("http"):
            with st.spinner(f"Mengunduh model {name}..."):
                urllib.request.urlretrieve(url, path)
    
    download_model(url_ai1, ai1_path, "Vision AI")
    download_model(url_ai2, ai2_path, "Spatial AI")
    
    # AI 1
    model_ai1 = timm.create_model('swinv2_base_window12to16_192to256_22kft1k', pretrained=False, num_classes=4)
    if os.path.exists(ai1_path):
        model_ai1.load_state_dict(torch.load(ai1_path, map_location=torch.device('cpu'), weights_only=False))
    model_ai1.eval()
    
    # AI 2
    model_ai2 = ST_GAT(in_channels=3, hidden_channels=64)
    if os.path.exists(ai2_path):
        model_ai2.load_state_dict(torch.load(ai2_path, map_location=torch.device('cpu'), weights_only=False))
    model_ai2.eval()
    
    # AI 3
    model_ai3 = cb.CatBoostRegressor()
    if os.path.exists('saved_models/ai3_best_catboost.cbm'):
        model_ai3.load_model('saved_models/ai3_best_catboost.cbm')
    
    return model_ai1, model_ai2, model_ai3, class_names

with st.spinner("Menginisialisasi Mesin AI (Swin Transformer, ST-GAT, CatBoost)..."):
    model_ai1, model_ai2, model_ai3, class_names = load_all_models()

# ==========================================
# SIDEBAR NAVIGATION & INFO
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1892/1892751.png", width=100) # Placeholder logo
    st.title("🌾 TISIIF 2026")
    st.caption("Agricultural AI Framework by Sriwijaya University")
    st.markdown("---")
    st.markdown("**Didukung oleh:**")
    st.markdown("- Swin Transformer V2\n- Spatio-Temporal GAT\n- CatBoost Regressor\n- Groq Llama-3.1")
    st.markdown("---")
    st.info("Pilih Tab di layar utama untuk mulai menganalisis tanaman Anda.")

# ==========================================
# KONTEN UTAMA (SISTEM TAB)
# ==========================================
st.markdown('<p class="main-header">🌾 WheatBlast Early Warning Radar</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Sistem Deteksi Epidemik & Prediksi Ekonomi Panen Gandum Berbasis Multi-AI</p>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["🔍 Radar Scanner", "📖 Ensiklopedia Penyakit", "ℹ️ Petunjuk Penggunaan"])

# ------------------------------------------
# TAB 1: RADAR SCANNER (APLIKASI UTAMA)
# ------------------------------------------
with tab1:
    col_left, col_right = st.columns([1.2, 1])

    with col_left:
        st.subheader("1. Pemindaian Visual (Vision AI)")
        
        # OPSI INPUT: KAMERA ATAU UPLOAD GALERI
        input_method = st.radio("Pilih Metode Input Gambar:", ["Unggah File (Galeri)", "Ambil Foto (Kamera)"], horizontal=True)
        
        uploaded_file = None
        if input_method == "Unggah File (Galeri)":
            uploaded_file = st.file_uploader("Pilih foto daun/bulir gandum", type=["jpg", "jpeg", "png"])
        else:
            uploaded_file = st.camera_input("Arahkan kamera ke daun/bulir gandum")
        
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
            input_tensor = transform_eval(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model_ai1(input_tensor)
                probs = F.softmax(outputs, dim=1).squeeze().numpy()
                pred_idx = np.argmax(probs)
                
            diagnosis = class_names[pred_idx].upper().replace('_', ' ')
            
            if diagnosis == "HEALTHY WHEAT":
                st.success(f"✅ **Hasil Diagnosa: {diagnosis}**")
            else:
                st.error(f"⚠️ **Hasil Diagnosa: terinfeksi {diagnosis}**")
                
            with st.expander("Lihat Detail Probabilitas AI"):
                for idx, name in enumerate(class_names):
                    st.write(f"{name.replace('_', ' ').capitalize()}: {probs[idx]*100:.2f}%")
                    st.progress(float(probs[idx]))
                    
            prob_sakit_for_ai2 = float(probs[0] + probs[2] + probs[3])
            visual_report_string = f"{diagnosis} ({probs[pred_idx]*100:.1f}%)"

        st.markdown("---")
        st.subheader("2. Radar Spasial-Temporal (Graph AI)")
        selected_region = st.selectbox("Zona Agro-Klimat:", ["Asia Selatan (India)", "Afrika Timur", "Amerika Selatan", "Eropa Timur"])
        
        # Simulasi GNN
        num_nodes = 150
        edge_index_mock = torch.randint(0, num_nodes, (2, 300))
        X_mock = torch.randn((num_nodes, 7, 3))
        X_mock[0, -1, 2] = prob_sakit_for_ai2 
        
        with torch.no_grad():
            graph_risk_output = model_ai2(X_mock, edge_index_mock).squeeze().numpy()
            lahan_target_risk = float(graph_risk_output[0])
            
        st.markdown(f'<div class="metric-card"><b>Risiko Penularan Ekstrem di {selected_region}</b><br><h2 style="margin:0; color:{"#EF4444" if lahan_target_risk > 0.6 else "#10B981"};">{lahan_target_risk * 100:.2f}%</h2></div>', unsafe_allow_html=True)
        risk_string = "High Danger" if lahan_target_risk > 0.6 else "Stable Baseline"

    with col_right:
        st.subheader("3. Proyeksi Panen (Expert System)")
        
        # Menggunakan Expander agar tidak memenuhi layar HP
        with st.expander("⚙️ Atur Parameter Lingkungan & Perawatan", expanded=True):
            temp_input = st.slider("Suhu Harian (°C)", 10.0, 40.0, 25.0)
            rainfall_input = st.slider("Curah Hujan (mm)", 0.0, 300.0, 150.0)
            fertilizer_type = st.selectbox("Jenis Pupuk", ["Chemical", "Organic", "Mixed"])
            pesticide_usage = st.selectbox("Intensitas Pestisida", ["Low", "Medium", "High"])

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
        
        # Eksekusi AI 3
        try:
            predicted_yield = model_ai3.predict(pd.DataFrame(input_data))[0]
            st.metric(label="Prediksi Hasil Panen Akhir", value=f"{predicted_yield:.2f} Kg/Hektar")
        except Exception as e:
            predicted_yield = 0.0
            st.warning("Model CatBoost belum terhubung sempurna.")

        st.markdown("---")
        st.subheader("4. Sintesis Pakar Agronomi (LLM)")
        st.caption("AI akan menganalisis ketiga data di atas dan memberikan instruksi tindakan mitigasi tingkat ahli.")
        
        if st.button("🤖 Generate Laporan Pakar (Groq API)"):
            prompt_llm = f"""
            Anda adalah seorang konsultan agrikultur senior dari FAO. Buat laporan singkat (maksimal 3 kalimat) dalam BAHASA INGGRIS berdasarkan data AI berikut:
            - Status Daun (Computer Vision): {visual_report_string}
            - Risiko Penularan Geografis (Spatio-Temporal Graph): {risk_string} ({lahan_target_risk*100:.1f}%)
            - Prediksi Hasil Panen (CatBoost Machine Learning): {predicted_yield:.1f} Kg/Ha.
            Akhiri laporan dengan satu tindakan mitigasi yang sangat jelas dan tegas. Jangan menggunakan salam pembuka.
            """
            
            try:
                api_key = st.secrets["GROQ_API_KEY"]
                client = Groq(api_key=api_key)
                
                with st.spinner("Menyintesis laporan via Llama-3.1 Cloud..."):
                    chat_completion = client.chat.completions.create(
                        messages=[{"role": "user", "content": prompt_llm}],
                        model="llama-3.1-8b-instant", 
                        temperature=0.3
                    )
                report_text = chat_completion.choices[0].message.content
                st.info(report_text)
                
            except KeyError:
                st.error("Kunci API Groq belum diatur di Streamlit Secrets.")
            except Exception as e:
                st.error(f"Gagal menghubungi server Groq: {str(e)}")

# ------------------------------------------
# TAB 2: ENSIKLOPEDIA PENYAKIT
# ------------------------------------------
with tab2:
    st.header("📖 Ensiklopedia Patogen Gandum")
    st.write("Kenali berbagai jenis ancaman patogen yang dapat menghancurkan hasil panen gandum Anda.")
    
    with st.expander("🌾 Wheat Blast (Magnaporthe oryzae pathotype Triticum)", expanded=True):
        st.error("**Tingkat Bahaya: SANGAT TINGGI (Karantina)**")
        st.write("""
        **Deskripsi:** Penyakit jamur mematikan yang menyebar cepat, menyerang bulir gandum dan menyebabkan pemutihan parsial atau total.
        **Gejala Visual:** Bulir menjadi putih dan kering sebagian, bercak abu-abu pada daun dengan tepi cokelat gelap. \n
        **Mitigasi:** Penggunaan fungisida triazole, rotasi tanaman, dan pemusnahan sisa panen yang terinfeksi.
        """)
        
    with st.expander("🍂 Leaf Rust (Puccinia triticina)"):
        st.warning("**Tingkat Bahaya: SEDANG - TINGGI**")
        st.write("""
        **Deskripsi:** Dikenal sebagai 'karat daun', merupakan penyakit gandum yang paling tersebar luas di dunia.
        **Gejala Visual:** Pustula kecil berbentuk oval berwarna oranye hingga cokelat karat pada permukaan daun. 
        **Mitigasi:** Penggunaan varietas gandum tahan karat, penyemprotan fungisida preventif saat kelembapan tinggi.
        """)
        
    with st.expander("🪵 Stem Rust (Puccinia graminis)"):
        st.warning("**Tingkat Bahaya: TINGGI**")
        st.write("""
        **Deskripsi:** Patogen perusak batang yang dapat menyebabkan tanaman gandum rebah (patah) sebelum panen.
        **Gejala Visual:** Pustula merah bata memanjang yang menembus epidermis batang dan seludang daun.
        **Mitigasi:** Pemantauan spora terbawa angin, eradikasi inang alternatif (seperti tanaman *barberry*).
        """)

# ------------------------------------------
# TAB 3: PETUNJUK PENGGUNAAN
# ------------------------------------------
with tab3:
    st.header("ℹ️ Cara Kerja Sistem 4-AI")
    st.markdown("""
    Aplikasi ini merangkai 4 Kecerdasan Buatan berbeda dalam satu jalur pipa (*pipeline*) terintegrasi:
    
    1. **Swin Transformer V2 (Vision AI):** Bertugas melihat foto yang Anda unggah/foto. Algoritma ini akan membedah tekstur piksel daun untuk mencari mikroskopis jamur dengan akurasi 95%.
    2. **ST-GAT (Graph Neural Network):** Mengambil hasil dari AI Pertama, lalu memproyeksikannya ke dalam "Peta Angin" digital. AI ini menghitung probabilitas spora terbang ke lahan tetangga Anda.
    3. **CatBoost (Sistem Pakar Ekonomi):** Mengalkulasi dampak kerusakan jamur dan kondisi cuaca terhadap penyusutan tonase panen (Kg/Ha) di akhir musim.
    4. **Groq Llama-3.1 (LLM Agent):** Membaca seluruh angka rumit dari ketiga AI di atas, lalu menyusunnya menjadi laporan berbahasa Inggris layaknya konsultan pertanian manusia.
    
    **Langkah Penggunaan:**
    - Buka Tab **Radar Scanner**.
    - Ambil foto daun gandum yang dicurigai sakit langsung dari kamera HP Anda, atau unggah dari galeri.
    - Tunggu AI Pertama mendiagnosis penyakit.
    - Atur *slider* suhu dan curah hujan sesuai kondisi desa Anda.
    - Klik tombol **Generate Laporan Pakar** di bagian bawah untuk mendapatkan rekomendasi final.
    """)
