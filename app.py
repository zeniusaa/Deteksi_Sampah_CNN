#
import streamlit as st
from PIL import Image
from fastai.vision.all import *
import pathlib
import os

# ✅ Patch untuk Windows
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Konfigurasi halaman
st.set_page_config(page_title="Deteksi Jenis Sampah", page_icon="🗑")
st.title("🗑 Deteksi Jenis Sampah")
st.write("Upload gambar sampah untuk diklasifikasikan dan dapatkan tips penanganannya.")

# ✅ Tips penanganan per jenis sampah
waste_handling_tips = {
    "cardboard": "📦 Kardus sebaiknya dipipihkan dan dikeringkan. Daur ulang atau digunakan kembali untuk kerajinan.",
    "glass": "🧪 Kaca dapat didaur ulang. Pisahkan berdasarkan warna, dan jangan dicampur dengan sampah organik.",
    "metal": "🔩 Logam seperti kaleng bisa dijual ke pengepul. Bersihkan dulu sebelum dikumpulkan.",
    "paper": "📄 Kertas bekas bisa didaur ulang. Jangan campur dengan sampah basah.",
    "plastic": "♻ Plastik bersih bisa didaur ulang. Hindari membakar karena menghasilkan racun.",
    "trash": "🚮 Sampah residu tidak bisa didaur ulang. Buang ke tempat sampah biasa atau TPA.",
}

# ✅ Cek apakah model tersedia
model_path = "data/best_model.pkl"
if not os.path.exists(model_path):
    st.error(f"❌ Model tidak ditemukan: {model_path}")
    st.stop()

# ✅ Load model
try:
    model = load_learner(model_path)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

# ✅ Upload gambar
uploaded_file = st.file_uploader("📷 Pilih gambar...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    if st.button("🔍 Deteksi Jenis Sampah"):
        try:
            img_fastai = PILImage.create(image)
            pred, pred_idx, probs = model.predict(img_fastai)

            # ✅ Tampilkan hasil prediksi
            st.subheader(f"Hasil Prediksi: `{pred}`")
            st.write(f"🎯 Probabilitas Tertinggi: {probs[pred_idx]:.4f}")

            # ✅ Detail probabilitas semua kelas
            st.subheader("🔎 Probabilitas Semua Kelas:")
            for label, prob in zip(model.dls.vocab, probs):
                st.write(f"- {label}: {prob:.4f}")

            # ✅ Tips penanganan
            st.subheader("🧾 Tips Penanganan:")
            tip = waste_handling_tips.get(str(pred).lower(), "⚠️ Tidak ada informasi penanganan untuk kategori ini.")
            st.info(tip)

        except Exception as e:
            st.error(f"❌ Terjadi kesalahan saat prediksi: {e}")

# ✅ Kembalikan patch jika perlu
pathlib.PosixPath = temp

