import streamlit as st
import torch
import pandas as pd
from transformers import AutoTokenizer
from model import IndoBERTHoaxClassifier

st.set_page_config(
    page_title="IndoBERT Hoax Detector",
    page_icon="🧠",
    layout="wide",
)

# LOAD MODEL & TOKENIZER
@st.cache_resource
def load_model():
    model = IndoBERTHoaxClassifier()
    model.load_state_dict(torch.load("hoax_detector_model.pth", map_location=torch.device("cpu")))
    model.eval()
    return model

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained("indobenchmark/indobert-base-p1")

model = load_model()
tokenizer = load_tokenizer()

st.markdown("""
<style>
    body {
        font-family: 'Inter', sans-serif;
        background-color: #f8fafc;
        color: #1e293b;
    }
    .user-bubble, .bot-bubble {
        padding: 1em 1.3em;
        border-radius: 14px;
        margin-bottom: 1em;
        max-width: 85%;
        line-height: 1.6em;
        word-wrap: break-word;
    }
    .user-bubble {
        background-color: #2563eb;
        color: white;
        align-self: flex-end;
        margin-left: auto;
    }
    .bot-bubble {
        background-color: #f1f5f9;
        color: #111827;
        border: 1px solid #e2e8f0;
    }
    .stTextArea textarea {
        border-radius: 10px !important;
    }
    .stButton > button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        border-radius: 10px;
        border: none;
        padding: 0.6em 1.4em;
    }
    .stButton > button:hover {
        background-color: #1d4ed8;
        transform: scale(1.03);
    }
    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 0.85em;
        margin-top: 2em;
    }
</style>
""", unsafe_allow_html=True)


# SIDEBAR

st.sidebar.title("🧠 IndoBERT Hoax Detector")
st.sidebar.caption("""
Aplikasi ini menggunakan model **IndoBERT**  
untuk mendeteksi berita **hoax** atau **valid**.
""")
page = st.sidebar.radio("Pilih Mode:", ["Chat Mode", "CSV Mode"])

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Cimanura Team**  
- Cindy Amelia Prameswari 
- Nur Salamah Azzahrah 
- Rachel Muthia Putri Nasty 
- Magrozan Qobus Zaidan 
""")

if page == "Chat Mode":

    st.markdown("<h1 style='text-align:center;'>Chat Mode — Deteksi Hoax</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#475569;'>Ketik berita dan biarkan AI menentukan apakah berita tersebut hoax atau valid.</p>", unsafe_allow_html=True)
    st.caption("<hr>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Riwayat Chat
    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='bot-bubble'>{msg['content']}</div>", unsafe_allow_html=True)

    # Input Chat
    user_input = st.text_area(
        "Ketik berita yang ingin diperiksa:",
        key="chat_input",
        height=120,
        placeholder="Contoh: Pemerintah memberikan bantuan langsung tunai sebesar 10 juta untuk semua warga..."
    )
    col1, col2, col3 = st.columns([1, 4, 3])
    with col1:
        submit = st.button("Kirim")
    with col2:
        clear = st.button("Hapus Chat")

    if clear:
        st.session_state.messages = []
        st.rerun()

    # Prediksi Chat
    if submit and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.spinner("Menganalisis berita..."):
            tokens = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                logits = model(**tokens)
                probs = torch.nn.functional.softmax(logits, dim=1)[0]
                pred_label = torch.argmax(probs).item()
                confidence = probs[pred_label].item() * 100

        label_map = {0: "✅ Berita Valid", 1: "🚫 Berita Hoax"}
        color = "#10b981" if pred_label == 0 else "#ef4444"
        response = f"<b style='color:{color};'>{label_map[pred_label]}</b><br>Tingkat keyakinan model: <b>{confidence:.2f}%</b>"

        st.session_state.messages.append({"role": "bot", "content": response})
        st.rerun()


elif page == "CSV Mode":
    st.markdown("<h1 style='text-align:center;'>CSV Mode — Deteksi Hoax </h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align:center;color:#475569;'>Unggah file CSV dan pilih kolom teks yang ingin diperiksa oleh model.</p>", unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Unggah file CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success(f"File berhasil dimuat! Jumlah data: {len(df)} baris.")

        # Pilihan kolom teks
        text_column = st.selectbox(
            "Pilih kolom teks untuk diprediksi:",
            options=df.columns.tolist(),
            help="Pilih kolom yang berisi teks berita atau artikel yang ingin diperiksa."
        )

        if st.button("Jalankan Prediksi"):
            with st.spinner("Sedang memproses semua teks..."):
                texts = df[text_column].astype(str).tolist()
                tokens = tokenizer(
                    texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128
                )
                with torch.no_grad():
                    logits = model(**tokens)
                    probs = torch.nn.functional.softmax(logits, dim=1)
                    preds = torch.argmax(probs, dim=1)

                df["prediction"] = preds.numpy()
                df["label"] = df["prediction"].map({0: "Berita Valid", 1: "Berita Hoax"})
                df["confidence (%)"] = probs.max(dim=1).values.numpy() * 100

            st.success("Analisis selesai! Berikut hasilnya:")
            st.dataframe(df[[text_column, "label", "confidence (%)"]])

            csv = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Unduh Hasil CSV",
                data=csv,
                file_name="hasil_prediksi_hoax.csv",
                mime="text/csv"
            )

            # Visualisasi hasil prediksi
            st.markdown("### 📈 Distribusi Hasil Prediksi")
            chart_data = df["label"].value_counts().reset_index()
            chart_data.columns = ["Kategori", "Jumlah"]
            st.bar_chart(chart_data.set_index("Kategori"))


st.markdown("""
<div class="footer">
Dikembangkan oleh <b>Magrozan Qobus Zaidan</b>
</div>
""", unsafe_allow_html=True)
