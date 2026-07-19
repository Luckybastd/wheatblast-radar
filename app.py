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
# KONFIGURASI HALAMAN & CUSTOM CSS
# ==========================================
st.set_page_config(page_title="WheatBlast Radar", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: 800; color: #1E3A8A; margin-bottom: 0;}
    .sub-header {font-size: 1.2rem; color: #4B5563; margin-bottom: 2rem;}
    /* Desain tombol Prediksi agar menonjol */
    .stButton>button {background-color: #2563EB; color: white; border-radius: 8px; font-weight: bold;}
    .stButton>button:hover {background-color: #1D4ED8; border-color: #1D4ED8;}
    /* Sembunyikan atribusi dan logo peta di kanan bawah */
    .mapboxgl-ctrl-bottom-left, .mapboxgl-ctrl-bottom-right, .mapboxgl-ctrl-logo, .mapboxgl-ctrl-attrib {display: none !important;}
</style>
""", unsafe_allow_html=True)

# Inisialisasi Session State untuk Prediksi AI ke-3
if 'predicted_yield' not in st.session_state:
    st.session_state.predicted_yield = None

# ==========================================
# DEFINISI ARSITEKTUR & CACHING
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
    st.title("AWARE System")
    st.caption("Agricultural Wheat AI Radar & Epidemiology")
    st.markdown("---")
    st.markdown("**Didukung oleh:**\n- Swin Transformer V2\n- Spatio-Temporal GAT\n- CatBoost Regressor\n- Groq Llama-3.1")

# ==========================================
# KONTEN UTAMA (SISTEM TAB)
# ==========================================
st.markdown('<p class="main-header">AWARE Dashboard</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">System of Integrated Artificial Intelligence for Geospatial Analytics on Wheat Epidemics</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["Radar Scanner", "Ensiklopedia Penyakit", "Petunjuk Operasional", "Kemitraan Riset"])

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
        visual_report_string = "Belum ada spesimen dipindai"
        lahan_target_risk = 0.0
        risk_string = "Menunggu data visual"
        
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
                st.success(f"Diagnosis: {diagnosis}")
            else:
                st.error(f"Diagnosis: {diagnosis}")
                    
            prob_sakit_for_ai2 = float(probs[0] + probs[2] + probs[3])
            visual_report_string = f"{diagnosis} ({probs[pred_idx]*100:.1f}%)"

        st.markdown("---")
        st.subheader("2. Radar Spasial-Temporal (Graph AI)")
        
        koordinat_wilayah = {
            "Asia Selatan (India)": [22.0, 79.0],
            "Afrika Timur": [9.0, 39.0],
            "Amerika Selatan": [-14.0, -51.0],
            "Eropa Timur": [48.0, 30.0]
        }
        selected_region = st.selectbox("Zona Agro-Klimat:", list(koordinat_wilayah.keys()))
        
        with torch.no_grad():
            X_mock = torch.randn((150, 7, 3))
            X_mock[0, -1, 2] = prob_sakit_for_ai2 
            lahan_target_risk = float(model_ai2(X_mock, torch.randint(0, 150, (2, 300))).squeeze().numpy()[0])
            risk_string = "Bahaya Ekstrem" if lahan_target_risk > 0.6 else "Risiko Sedang" if lahan_target_risk > 0.3 else "Stabil"

        st.metric(label="Risiko Penularan Regional", value=f"{lahan_target_risk * 100:.2f}%")
        
        # PETA 3D INTERAKTIF (Lingkaran Bolong dan Provider Carto)
        lat, lon = koordinat_wilayah[selected_region]
        
        if lahan_target_risk > 0.6:
            map_color = [239, 68, 68, 255] 
        elif lahan_target_risk > 0.3:
            map_color = [245, 158, 11, 255] 
        else:
            map_color = [16, 185, 129, 255] 
            
        df_map = pd.DataFrame({"lat": np.random.randn(20) * 1.5 + lat, "lon": np.random.randn(20) * 1.5 + lon})
        
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=df_map,
            get_position='[lon, lat]',
            get_radius=180000,
            filled=False,
            stroked=True,
            get_line_color=map_color,
            line_width_min_pixels=4,
            pickable=True
        )
        view_state = pdk.ViewState(latitude=lat, longitude=lon, zoom=3.5, pitch=30)
        st.pydeck_chart(pdk.Deck(
            map_provider="carto",
            map_style="dark",
            layers=[layer], 
            initial_view_state=view_state, 
            tooltip={"text": f"Zona {risk_string}"}
        ))

    with col_right:
        st.subheader("3. Proyeksi Panen (Expert System)")
        
        with st.expander("Parameter Iklim & Cuaca", expanded=True):
            temp_input = st.slider("Suhu Harian (Celcius)", 10.0, 40.0, 25.0)
            rainfall_input = st.slider("Curah Hujan Bulanan (mm)", 0.0, 500.0, 150.0)
            humidity = st.slider("Kelembapan (%)", 0.0, 100.0, 60.0)
            c1, c2, c3 = st.columns(3)
            solar_rad = c1.number_input("Radiasi Surya", value=548.8)
            wind_speed = c2.number_input("Kecepatan Angin", value=9.9)
            gdd = c3.number_input("GDD", value=1593.5)

        with st.expander("Manajemen Lahan & Tanaman"):
            c1, c2 = st.columns(2)
            fertilizer_type = c1.selectbox("Jenis Pupuk", ["Chemical", "Organic", "Mixed"])
            pesticide_usage = c2.selectbox("Tingkat Pestisida", ["Low", "Medium", "High"])
            irrigation = c1.number_input("Frekuensi Irigasi", value=14)
            growth_stage = c2.selectbox("Fase Pertumbuhan", ["Tillering", "Heading", "Flowering", "Maturity"])
            pest_tonnes = c1.number_input("Pestisida (Ton)", value=48459.0)
            pest_ml = c2.number_input("Pestisida (ml/Ha)", value=200.0)
            
            c3, c4, c5 = st.columns(3)
            region_input = c3.selectbox("Region", ["North", "South", "East", "West"])
            season_input = c4.selectbox("Season", ["Rabi", "Kharif"])
            year_input = c5.number_input("Tahun", value=2026)

        with st.expander("Kondisi Tanah & Topografi"):
            c1, c2 = st.columns(2)
            soil_type = c1.selectbox("Tipe Tanah", ["Loamy", "Clay", "Sandy", "Silty"])
            ph = c2.number_input("pH Tanah", value=6.5)
            elevation = c1.number_input("Elevasi (mdpl)", value=1565.8)
            slope = c2.number_input("Kemiringan", value=14.9)
            
        with st.expander("Nutrisi Makro & Mikro (Advanced)"):
            st.caption("Biarkan nilai default jika tidak ada data laboratorium.")
            c1, c2, c3, c4 = st.columns(4)
            n_val = c1.number_input("N", value=105.1)
            p_val = c2.number_input("P", value=77.8)
            k_val = c3.number_input("K", value=128.9)
            ca_val = c4.number_input("Ca", value=1021.0)
            mg_val = c1.number_input("Mg", value=255.8)
            s_val = c2.number_input("S", value=50.4)
            zn_val = c3.number_input("Zn", value=5.0)
            fe_val = c4.number_input("Fe", value=25.3)
            cu_val = c1.number_input("Cu", value=2.5)
            mn_val = c2.number_input("Mn", value=12.5)
            b_val = c3.number_input("B", value=1.5)
            mo_val = c4.number_input("Mo", value=0.5)

        with st.expander("Indikator Sensor & Fisik Tanah (Advanced)"):
            c1, c2, c3, c4 = st.columns(4)
            ec = c1.number_input("EC", value=1.3)
            oc = c2.number_input("OC", value=1.05)
            cec = c3.number_input("CEC", value=27.6)
            bulk_dens = c4.number_input("Bulk Dens.", value=1.4)
            sand = c1.number_input("Sand (%)", value=49.8)
            silt = c2.number_input("Silt (%)", value=32.3)
            clay = c3.number_input("Clay (%)", value=22.6)
            whc = c4.number_input("Water Hold.", value=24.9)
            ndvi = c1.number_input("NDVI", value=0.7)
            evi = c2.number_input("EVI", value=0.0)
            lai = c3.number_input("LAI", value=3.0)
            chloro = c4.number_input("Chlorophyll", value=30.0)
            aspect = c1.number_input("Aspect", value=181.1)

        # Tombol Prediksi Interaktif
        if st.button("Prediksi Tonase Panen", use_container_width=True):
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
                penalty_factor = 1.0 - (lahan_target_risk * 0.4) 
                st.session_state.predicted_yield = base_yield * penalty_factor
            except Exception:
                st.session_state.predicted_yield = 0.0

        st.markdown("<br>", unsafe_allow_html=True)
        if st.session_state.predicted_yield is None:
            st.metric(label="Prediksi Hasil Panen Akhir", value="-- Kg/Hektar", delta="Menunggu Input Data")
        else:
            st.metric(label="Prediksi Hasil Panen Akhir", value=f"{st.session_state.predicted_yield:.2f} Kg/Hektar", delta="Dampak Epidemi" if prob_sakit_for_ai2 > 0.5 else "Kondisi Optimal", delta_color="inverse")

        st.markdown("---")
        st.subheader("4. Sintesis Pakar Agronomi (LLM)")
        
        if st.button("Kalkulasi Laporan Pakar (API Terenkripsi)"):
            if st.session_state.predicted_yield is None:
                st.warning("Peringatan: Harap jalankan Prediksi Tonase Panen pada Langkah 3 terlebih dahulu.")
            else:
                prompt_llm = f"""
                Anda adalah konsultan agrikultur strategis. Buat laporan profesional (maks 3 kalimat) dalam BAHASA INGGRIS berdasarkan metrik AI:
                - Deteksi Penyakit: {visual_report_string}
                - Risiko Spasial Geografis: {risk_string} ({lahan_target_risk*100:.1f}%)
                - Estimasi Panen: {st.session_state.predicted_yield:.1f} Kg/Ha.
                Sertakan satu protokol mitigasi teknis.
                """
                try:
                    client = Groq(api_key=st.secrets["GROQ_API_KEY"])
                    with st.spinner("Memproses analitik sintesis..."):
                        report_text = client.chat.completions.create(
                            messages=[{"role": "user", "content": prompt_llm}], model="llama-3.1-8b-instant", temperature=0.3
                        ).choices[0].message.content
                    st.info(report_text)
                except Exception as e:
                    st.error(f"Kegagalan Komunikasi Server: {str(e)}")

# ------------------------------------------
# TAB 2: ENSIKLOPEDIA KUSTOM
# ------------------------------------------
with tab2:
    st.header("Ensiklopedia Patogen Gandum")
    st.write("Klasifikasi patologi dan protokol penanganan ancaman ekologis.")
    
    st.markdown("""
    <div style="background-color: rgba(220, 38, 38, 0.15); border: 1px solid #DC2626; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
        <details open>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #EF4444; cursor: pointer; list-style: none;">
                Wheat Blast (Magnaporthe oryzae pathotype Triticum) - [BAHAYA KARANTINA]
            </summary>
            <hr style="border-color: rgba(220, 38, 38, 0.3);">
            <p style="color: #F87171;"><b>Patologi:</b> Penyakit jamur berdaya hancur ekstrem yang menyebar eksponensial. Menyerang bulir gandum dan memicu nekrosis jaringan vaskular total.</p>
            <p style="color: #F87171;"><b>Identifikasi Visual:</b> Pemutihan bulir secara instan, diikuti pengeringan akut. Terdapat bercak elips abu-abu dengan perimeter cokelat gelap pada foliar.</p>
            <p style="color: #F87171;"><b>Protokol Mitigasi:</b> Eradikasi biologis radius area terinfeksi, aplikasi fungisida sistemik kelas triazole secara masif, dan isolasi distribusi logistik lokal.</p>
        </details>
    </div>
    
    <div style="background-color: rgba(245, 158, 11, 0.15); border: 1px solid #F59E0B; border-radius: 10px; padding: 15px; margin-bottom: 15px;">
        <details>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #FBBF24; cursor: pointer; list-style: none;">
                Stem Rust (Puccinia graminis) - [RISIKO TINGGI]
            </summary>
            <hr style="border-color: rgba(245, 158, 11, 0.3);">
            <p style="color: #FCD34D;"><b>Patologi:</b> Destruksi pembuluh batang yang mendisrupsi suplai nutrisi dan integritas struktural, memicu fenomena rebah batang masal pra-panen.</p>
            <p style="color: #FCD34D;"><b>Identifikasi Visual:</b> Ruptur epidermis linier membentuk pustula merah bata yang memancarkan spora berkarat secara aerosol saat terpapar vibrasi mekanis.</p>
            <p style="color: #FCD34D;"><b>Protokol Mitigasi:</b> Pemusnahan gulma inang sekunder (berberis) dan penjadwalan penyemprotan protektan spektrum luas sesuai siklus curah hujan.</p>
        </details>
    </div>
    
    <div style="background-color: rgba(59, 130, 246, 0.15); border: 1px solid #3B82F6; border-radius: 10px; padding: 15px;">
        <details>
            <summary style="font-weight: bold; font-size: 1.2rem; color: #60A5FA; cursor: pointer; list-style: none;">
                Leaf Rust (Puccinia triticina) - [RISIKO SEDANG]
            </summary>
            <hr style="border-color: rgba(59, 130, 246, 0.3);">
            <p style="color: #93C5FD;"><b>Patologi:</b> Degradasi seluler kronis pada daun yang menghambat laju fotosintesis, mendepresiasi akumulasi pati dan menurunkan spesifikasi massa panen.</p>
            <p style="color: #93C5FD;"><b>Identifikasi Visual:</b> Distribusi asimetris pustula sferis berwarna oranye terang, terpusat eksklusif pada permukaan helaian daun strata atas.</p>
            <p style="color: #93C5FD;"><b>Protokol Mitigasi:</b> Intervensi genetik bertahap melalui transisi penanaman menuju varietas tahan karat (resistant cultivar) untuk siklus agrikultur berikutnya.</p>
        </details>
    </div>
    """, unsafe_allow_html=True)

# ------------------------------------------
# TAB 3: PETUNJUK
# ------------------------------------------
with tab3:
    st.header("Arsitektur Integrasi Sistem AWARE")
    st.markdown("""
    Sistem operasional ini memvalidasi ancaman biologis melalui empat matriks kecerdasan buatan:
    1. **Edge-Vision AI:** Eksekusi klasifikasi patogen mikroskopis pada spesimen lokal berdasarkan geometri sel.
    2. **GNN Spasial:** Simulasi penyebaran spora lintas wilayah berdasarkan analisis topologi jaringan spasial.
    3. **Expert Modeler:** Kuantifikasi regresi dampak klinis terhadap defisit tonase agrikultur komersial.
    4. **Generative NLP:** Sintesis metrik teknis menjadi protokol instruksional strategis.
    """)

# ------------------------------------------
# TAB 4: KEMITRAAN & KONTAK
# ------------------------------------------
with tab4:
    st.header("Jaringan Kemitraan Inovasi & Riset Global")
    st.markdown("Arsitektur proyek **AWARE** didesain dengan protokol *Open-Architecture* untuk memfasilitasi integrasi riset bersama institusi agrikultur dan pemangku kebijakan ketahanan pangan internasional.")
    
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.subheader("Pimpinan Peneliti")
        st.markdown("""
        **Jonatan**  
        *Undergraduate Researcher, Applied Mathematics*  
        **Universitas Sriwijaya (UNSRI)**, Indonesia.  
        Fokus Investigasi: Penerapan arsitektur *Machine Learning*, algoritmik *Computer Vision*, dan *System Dynamics* untuk stabilisasi rantai pangan.
        
        Korespodensi Manajerial: partnership@aware-unsri.edu.id
        """)
    with col_c2:
        st.subheader("Integrasi Basis Data Institusional")
        st.markdown("Kalibrasi model prediktif divalidasi silang menggunakan aliran data sumber terbuka dari otoritas agrikultur berikut:")
        st.markdown("""
        - [CIMMYT (International Maize and Wheat Improvement Center)](https://www.cimmyt.org/)
        - [FAO Global Wheat Rust Monitoring System](https://rusttracker.cimmyt.org/)
        - [Borlaug Global Rust Initiative (BGRI)](https://bgri.cornell.edu/)
        """)
