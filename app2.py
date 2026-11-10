import hashlib
import re
import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import base64
import json
import os
import requests
from io import BytesIO
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer



# ── DATABASE INIT ──────────────────────────────────────────────────
from database import (
    init_db, get_all_produk, insert_produk,
    update_produk, delete_produk, sync_to_csv, import_from_csv
)

# ===============================
# KONFIGURASI
# ===============================
GEMINI_API_KEY      = "
GEMINI_URL          = ""
MODEL_PATH          = "Model2.pt"
CSV_FILES           = ["carimanis_botol.csv", "carimanis_kaleng.csv"]
AKG_GULA            = 50
YOLO_CONF_THRESHOLD = 0.50
STABILIZE_FRAMES    = 10

# Pastikan tabel ada setiap kali app dijalankan
init_db()

# Seed sekali: jika DB masih kosong, impor dari CSV lama yang sudah ada
if get_all_produk().empty:
    for csv_f in CSV_FILES:
        n = import_from_csv(csv_f)
        if n > 0:
            print(f"[DB] Seed {n} produk dari {csv_f}")

# ===============================
# SUGAR DATABASE — label YOLO
# ===============================
SUGAR_DB = {
    "Ultra Milk FC":              4.00,   # 8g / 200ml * 100
    "Pocari Sweat":               2.57,   # 9g / 350ml * 100
    "Teh Pucuk Lass Sugar":       3.43,   # 12g / 350ml * 100
    "Milo":                       8.33,   # 15g / 180ml * 100
    "Teh Kotak":                  5.67,   # 17g / 300ml * 100
    "Teh Pucuk":                  5.14,   # 18g / 350ml * 100
    "C1000":                      9.29,   # 13g / 140ml * 100
    "Nipis Madu":                 6.67,   # 22g / 330ml * 100
    "Coca Cola":                 10.80,   # 27g / 250ml * 100
}

PRODUCT_META = {
    "Ultra Milk FC":             {"netto": 200, "kalori": 120, "jenis": "Susu cair"},
    "Pocari Sweat":              {"netto": 350, "kalori": 35,  "jenis": "Isotonik"},
    "Teh Pucuk Lass Sugar":      {"netto": 350, "kalori": 45,  "jenis": "Teh siap minum"},
    "Milo":                      {"netto": 180, "kalori": 35,  "jenis": "Susu kemasan"},
    "Teh Kotak":                 {"netto": 300, "kalori": 70,  "jenis": "Teh siap minum"},
    "Teh Pucuk":                 {"netto": 350, "kalori": 70,  "jenis": "Teh siap minum"},
    "C1000":                     {"netto": 140, "kalori": 50,  "jenis": "Minuman vitamin"},
    "Nipis Madu":                {"netto": 330, "kalori": 110, "jenis": "Minuman Soda"},
    "Coca Cola":                 {"netto": 250, "kalori": 100, "jenis": "Minuman bersoda"},
}
# ===============================
# NUTRI-LEVEL CONFIG
# ===============================
NUTRI_LEVEL_CONFIG = {
    "A": {
        "label":       "A",
        "warna":       "Hijau",
        "kategori":    "Pilihan Lebih Sehat",
        "hex":         "#00A651",
        "cv2_bgr":     (81, 166, 0),
        "deskripsi":   "Kandungan gula sangat rendah sehingga lebih aman untuk konsumsi harian.",
        "rekomendasi": "Dapat dikonsumsi lebih sering sebagai pilihan minuman yang lebih sehat.",
        "emoji":       "🟢",
        "range":       "≤ 1 g / 100 ml",
    },

    "B": {
        "label":       "B",
        "warna":       "Hijau Muda",
        "kategori":    "Rendah Gula",
        "hex":         "#8BC53F",
        "cv2_bgr":     (63, 197, 139),
        "deskripsi":   "Kandungan gula rendah dan masih tergolong baik untuk dikonsumsi.",
        "rekomendasi": "Konsumsi tetap disarankan dalam batas kebutuhan harian.",
        "emoji":       "🟩",
        "range":       "> 1 – 5 g / 100 ml",
    },

    "C": {
        "label":       "C",
        "warna":       "Oranye",
        "kategori":    "Sedang",
        "hex":         "#F9A01B",
        "cv2_bgr":     (27, 160, 249),
        "deskripsi":   "Kandungan gula sedang dan perlu diperhatikan dalam konsumsi harian.",
        "rekomendasi": "Batasi konsumsi agar asupan gula harian tidak berlebihan.",
        "emoji":       "🟠",
        "range":       "> 5 – 10 g / 100 ml",
    },

    "D": {
        "label":       "D",
        "warna":       "Merah",
        "kategori":    "Tinggi Gula",
        "hex":         "#ED1C24",
        "cv2_bgr":     (36, 28, 237),
        "deskripsi":   "Kandungan gula tinggi dan berisiko jika dikonsumsi terlalu sering.",
        "rekomendasi": "Disarankan membatasi konsumsi untuk menjaga kesehatan.",
        "emoji":       "🔴",
        "range":       "> 10 g / 100 ml",
    },
}



def get_nutri_level(gula: float) -> tuple:
    if gula <= 1:    return "A", NUTRI_LEVEL_CONFIG["A"]
    elif gula <= 5:  return "B", NUTRI_LEVEL_CONFIG["B"]
    elif gula <= 10: return "C", NUTRI_LEVEL_CONFIG["C"]
    else:            return "D", NUTRI_LEVEL_CONFIG["D"]


def classify_sugar(value: float) -> tuple[str, str]:
    level, cfg = get_nutri_level(value)
    return level, cfg["hex"]


def get_product_info(name: str) -> dict:
    gula = SUGAR_DB.get(name, 0.0)
    meta = PRODUCT_META.get(name, {"netto": 250, "kalori": 0, "jenis": "Minuman Kemasan"})
    total_gula = round(gula * meta["netto"] / 100, 1)
    akg = round((total_gula / AKG_GULA) * 100, 1)
    lv_kode, lv_cfg = get_nutri_level(gula)
    return {
        "name": name, "gula_per_100ml": gula, "total_gula": total_gula,
        "netto": meta["netto"], "kalori": meta["kalori"], "jenis": meta["jenis"],
        "level_kode": lv_kode, "level_cfg": lv_cfg, "akg": akg,
    }


def build_manual_product_result(name: str, source: str = "Manual", confidence: float = 0.0, all_similar=None, csv_df: pd.DataFrame | None = None) -> dict:
    if csv_df is not None and not csv_df.empty and "name" in csv_df.columns:
        try:
            match_mask = csv_df["name"].astype(str).str.strip().eq(str(name).strip())
            if match_mask.any():
                row = csv_df.loc[match_mask.idxmax()]
                gula = float(row.get("gula_per_100ml", 0.0) or 0.0)
                total_gula = float(row.get("total_gula", 0.0) or 0.0)
                kalori = float(row.get("kalori", 0.0) or 0.0)
                netto = float(row.get("netto", 250.0) or 250.0)
            else:
                raise LookupError
        except Exception:
            info = get_product_info(name)
            gula = float(info["gula_per_100ml"])
            total_gula = float(info["total_gula"])
            kalori = float(info["kalori"])
            netto = float(info.get("netto", 250.0))
    else:
        info = get_product_info(name)
        gula = float(info["gula_per_100ml"])
        total_gula = float(info["total_gula"])
        kalori = float(info["kalori"])
        netto = float(info.get("netto", 250.0))

    akg = round((total_gula / AKG_GULA) * 100, 1)
    level_kode, level_cfg = get_nutri_level(gula)
    result = {
        "name": name,
        "gula": gula,
        "gula_per_100ml": gula,
        "total_gula": total_gula,
        "akg": akg,
        "kategori": level_cfg["label"],
        "level_kode": level_kode,
        "level_cfg": level_cfg,
        "kalori": kalori,
        "jenis": "Minuman Kemasan",
        "confidence": confidence,
        "source": source,
        "netto": netto,
    }
    if all_similar is not None:
        result["all_similar"] = all_similar
    return result

# ===============================
# LOAD CSV DATABASE (Vector DB)
# ===============================
# HAPUS versi lama, ganti dengan ini:
def load_csv_db_with_watch() -> pd.DataFrame:
    """Baca dari SQLite dengan cache invalidation berdasarkan file mtime."""
    import os
    db_file = "database.db"
    
    # Dapatkan mtime file database (kapan terakhir diubah)
    try:
        db_mtime = os.path.getmtime(db_file) if os.path.exists(db_file) else 0
    except:
        db_mtime = 0
    
    # Gunakan mtime sebagai cache key untuk invalidate cache otomatis
    @st.cache_data(show_spinner="Memuat data produk...", ttl=None)
    def _load_with_key(_mtime):
        """Baca dari SQLite (bukan langsung dari CSV)."""
        df = get_all_produk()
        if df.empty:
            return pd.DataFrame()
        df["total_gula"]     = pd.to_numeric(df["total_gula"],     errors="coerce").fillna(0)
        df["netto"]          = pd.to_numeric(df["netto"],           errors="coerce").fillna(250)
        df["kalori"]         = pd.to_numeric(df["kalori"],          errors="coerce").fillna(0)
        df["gula_per_100ml"] = df.apply(
            lambda r: round((r["total_gula"] / r["netto"]) * 100, 2)
            if r["netto"] > 0 else 0, axis=1
        )
        return df
    
    return _load_with_key(db_mtime)

@st.cache_data
def load_csv_db() -> pd.DataFrame:
    """Wrapper untuk load_csv_db_with_watch."""
    return load_csv_db_with_watch()


@st.cache_resource
def build_vector_db(_df: pd.DataFrame):
    if _df.empty:
        return None, None, None
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    names    = _df["name"].tolist()
    vectors  = embedder.encode(names, show_progress_bar=False)
    return embedder, np.array(vectors), names

def find_similar(query: str, embedder, vectors, names, df, top_k=5, min_similarity=0.0):
    """
    Cari produk di vector DB yang paling mirip dengan query menggunakan semantic similarity.
    
    Args:
        query: Query string (e.g., "Coca Cola Original")
        embedder: SentenceTransformer embedding model
        vectors: numpy array of product embeddings
        names: list of product names
        df: DataFrame berisi product info (gula_per_100ml, total_gula, netto, kalori)
        top_k: Jumlah hasil top yang dikembalikan
        min_similarity: Minimum similarity threshold (0.0-1.0)
    
    Returns:
        list of dicts dengan keys: name, gula_per_100ml, total_gula, netto, kalori, packaging, similarity
    """
    q_vec = embedder.encode([query])
    sims  = cosine_similarity(q_vec, vectors)[0]
    top_idx = sims.argsort()[::-1][:top_k]
    results = []
    for i in top_idx:
        sim_score = float(sims[i])
        # Filter by minimum similarity threshold
        if sim_score < min_similarity:
            continue
        row = df.iloc[i]
        results.append({
            "name":          row["name"],
            "gula_per_100ml": row["gula_per_100ml"],
            "total_gula":    row["total_gula"],
            "netto":         row["netto"],
            "kalori":        row["kalori"],
            "packaging":     row.get("packaging", "-"),
            "similarity":    round(sim_score * 100, 1),
        })
    return results

def render_nearest_comparison(
    target_name: str,
    target_gula: float,
    comp_df: pd.DataFrame,
    top_k: int = 5
):
    if comp_df.empty:
        return

    st.subheader("🔍 Produk Kandungan Gula Terdekat")

    df = comp_df.copy()
    df["Selisih"] = (df["gula_per_100ml"] - target_gula).abs()
    df = df.sort_values(["Selisih", "name"]).head(top_k).copy()

    LEVEL_LABEL = {
        "A": "Lebih Sehat",
        "B": "Rendah Gula",
        "C": "Sedang",
        "D": "Tinggi Gula"
    }

    max_gula = max(df["gula_per_100ml"].max(), 1)

    medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

    for idx, (_, row) in enumerate(df.iterrows()):

        lv = row["level"]
        gula_val = float(row["gula_per_100ml"])

        is_target = row["name"] == target_name

        if lv == "A":
            icon = "🟢"
        elif lv == "B":
            icon = "🟩"
        elif lv == "C":
            icon = "🟨"
        else:
            icon = "🟥"

        if is_target:
            status = "⭐ Produk Ini"
        elif gula_val < target_gula:
            status = "✅ Lebih Sehat"
        elif gula_val > target_gula:
            status = "⚠️ Lebih Tinggi"
        else:
            status = "➖ Sama"

        with st.container(border=True):

            col1, col2 = st.columns([4, 1])

            with col1:
                st.markdown(
                    f"#### {medals[idx] if idx < len(medals) else ''} {row['name']}"
                )

            with col2:
                if is_target:
                    st.success("Target")
                else:
                    st.info(status)

            st.markdown(
                f"{icon} **Level {lv} — {LEVEL_LABEL.get(lv, lv)}**"
            )

            progress = gula_val / max_gula
            st.progress(progress)

            st.caption(
                f"Gula per 100 ml: {gula_val:.1f} g"
            )

            c1, c2, c3 = st.columns(3)

            with c1:
                st.metric(
                    "Kalori",
                    f"{float(row['kalori']):.0f} kkal"
                )

            with c2:
                st.metric(
                    "Total Gula",
                    f"{float(row['total_gula']):.1f} g"
                )

            with c3:

                if is_target:
                    st.metric(
                        "Selisih",
                        "-"
                    )
                else:
                    selisih = gula_val - target_gula

                    st.metric(
                        "Selisih",
                        f"{selisih:+.2f} g"
                    )

    # Ringkasan terbaik & terburuk
    st.divider()

    best_row = df.loc[df["gula_per_100ml"].idxmin()]
    worst_row = df.loc[df["gula_per_100ml"].idxmax()]

    col_best, col_worst = st.columns(2)

    with col_best:
        st.success(
            f"💚 Paling Rendah Gula\n\n"
            f"**{best_row['name']}**\n\n"
            f"{best_row['gula_per_100ml']:.1f} g/100ml • Level {best_row['level']}"
        )

    with col_worst:
        st.error(
            f"🔴 Paling Tinggi Gula\n\n"
            f"**{worst_row['name']}**\n\n"
            f"{worst_row['gula_per_100ml']:.1f} g/100ml • Level {worst_row['level']}"
        )


def render_upload_analysis(found_product: dict, similar: list[dict] | None = None):
    st.subheader(f"🥇 Produk Terdeteksi: {found_product['name']}")

    render_product_card(
        found_product,
        source=found_product.get("source", "Upload")
    )

    if similar:

        st.divider()
        st.subheader("🔗 Top 5 Produk Paling Mirip")

        LEVEL_LABEL = {
            "A": "Lebih Sehat",
            "B": "Rendah Gula",
            "C": "Sedang",
            "D": "Tinggi Gula"
        }

        target_gula = float(found_product.get("gula", 0))

        max_gula = max(
            [float(r["gula_per_100ml"]) for r in similar]
        )

        medals = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"]

        for i, r in enumerate(similar):

            lv_kode, _ = get_nutri_level(r["gula_per_100ml"])

            gula_val = float(r["gula_per_100ml"])
            sim_pct = float(r.get("similarity", 0))

            if lv_kode == "A":
                level_icon = "🟢"
            elif lv_kode == "B":
                level_icon = "🟩"
            elif lv_kode == "C":
                level_icon = "🟨"
            else:
                level_icon = "🟥"

            if i == 0:
                status = "⭐ Produk Terdeteksi"
            elif gula_val < target_gula:
                status = "✅ Lebih Sehat"
            elif gula_val > target_gula:
                status = "⚠️ Gula Lebih Tinggi"
            else:
                status = "➖ Setara"

            with st.container(border=True):

                col1, col2 = st.columns([4, 1])

                with col1:
                    st.markdown(
                        f"### {medals[i] if i < len(medals) else ''} {r['name']}"
                    )

                with col2:
                    st.metric(
                        "Mirip",
                        f"{sim_pct:.1f}%"
                    )

                st.markdown(
                    f"**{level_icon} Level {lv_kode} — {LEVEL_LABEL.get(lv_kode)}**"
                )

                st.caption(status)

                progress_value = (
                    gula_val / max_gula if max_gula > 0 else 0
                )

                st.progress(progress_value)

                st.caption(
                    f"Gula per 100 ml: {gula_val:.1f} g"
                )

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Kalori",
                        f"{float(r['kalori']):.0f} kkal"
                    )

                with c2:
                    st.metric(
                        "Total Gula",
                        f"{float(r['total_gula']):.1f} g"
                    )

                with c3:

                    if i == 0:
                        st.metric(
                            "Status",
                            "Produk Ini"
                        )
                    else:
                        selisih = gula_val - target_gula

                        st.metric(
                            "Selisih",
                            f"{selisih:+.1f} g"
                        )

    else:
        render_nearest_comparison(
            found_product["name"],
            found_product["gula"],
            comp_df,
            top_k=5
        )

def render_summary_panel(found_product: dict | None = None):

    if found_product is None:
        return

    confidence = found_product.get("confidence")
    label = found_product["level_kode"]
    cfg = found_product["level_cfg"]

    badge_color = cfg["hex"]
    conf_str = f"{confidence:.0%}" if confidence is not None else "–"

    st.markdown(f"""
    <div class="ns-summary" style="border-color:rgba(0,0,0,0.08);">
        <div class="ns-summary-title">Ringkasan Deteksi</div>
        <div class="ns-summary-grid">
            <div class="ns-summary-item">
                <strong>Produk</strong>
                <span style="font-size:15px;">{found_product['name']}</span>
            </div>
            <div class="ns-summary-item">
                <strong>Gula / 100ml</strong>
                <span style="font-size:18px;">{found_product['gula']} <small style="font-size:12px;font-weight:500;opacity:.6">g</small></span>
            </div>
            <div class="ns-summary-item">
                <strong>Nutri-Level</strong>
                <span>
                    <span style="display:inline-block;padding:5px 14px;border-radius:99px;
                        background:{badge_color};color:white;font-size:13px;font-weight:800;
                        box-shadow:0 3px 10px {badge_color}55;letter-spacing:0.3px;">
                        {cfg['emoji']} Level {label}
                    </span>
                </span>
            </div>
            <div class="ns-summary-item">
                <strong>Confidence</strong>
                <span style="font-size:16px;">{conf_str}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

@st.cache_data
def build_comparison_db(_csv_df: pd.DataFrame) -> pd.DataFrame:
    rows_yolo = []
    for name, gula in SUGAR_DB.items():
        meta = PRODUCT_META.get(name, {"netto": 250, "kalori": 0, "jenis": "-"})
        lv_kode, _ = get_nutri_level(gula)
        total_gula = round(gula * meta["netto"] / 100, 1)
        rows_yolo.append({
            "name": name, "gula_per_100ml": gula,
            "kalori": meta["kalori"], "netto": meta["netto"],
            "total_gula": total_gula,
            "level": lv_kode, "source": "YOLO-DB",
        })
    df_yolo = pd.DataFrame(rows_yolo)

    if not _csv_df.empty:
        df_csv = _csv_df[["name", "gula_per_100ml", "kalori", "netto", "total_gula"]].copy()
        df_csv["level"]  = df_csv["gula_per_100ml"].apply(lambda g: get_nutri_level(g)[0])
        df_csv["source"] = "CSV-DB"
    else:
        df_csv = pd.DataFrame(columns=df_yolo.columns)

    combined = pd.concat([df_yolo, df_csv], ignore_index=True)
    combined = combined.drop_duplicates(subset=["name"], keep="first")
    combined = combined.sort_values("gula_per_100ml").reset_index(drop=True)
    return combined

# ===============================
# GEMINI VISION
# ─────────────────────────────────────────────────────────────────────
# Menggunakan pendekatan dari app.py:
#   • Prompt menyebut daftar produk SUGAR_DB → Gemini langsung pilih dari daftar
#   • Regex fallback untuk ambil JSON jika ada teks tambahan
#   • Kompresi gambar + MD5 cache dari appfinal.py → hemat quota
# ===============================

def _compress_image(image_pil: Image.Image, max_size: int = 768) -> tuple[bytes, str]:
    """Kompres & resize gambar, kembalikan (bytes, md5_hash)."""
    img = image_pil.copy()
    img.thumbnail((max_size, max_size))
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=82, optimize=True)
    img_bytes = buf.getvalue()
    return img_bytes, hashlib.md5(img_bytes).hexdigest()


@st.cache_data(show_spinner=False, ttl=3600)
def _cached_gemini_call(img_hash: str, img_b64: str) -> dict:
    """
    Inti request ke Gemini API.
    Di-cache berdasarkan hash gambar — gambar sama tidak dikirim ulang.
    Menggunakan prompt yang lebih sederhana dan valid seperti appfinal.py.
    """
    prompt = """
Kamu adalah sistem pengenal produk minuman kemasan Indonesia untuk database NutriScan.
Tugasmu adalah menganalisis gambar produk dan mengekstrak data agar sesuai dengan standar format database.

### ATURAN UTAMA IDENTIFIKASI:
1. Nama Produk & Netto: Harus dapat dipastikan dari gambar dengan tingkat keyakinan tinggi.
2. Ukuran Netto/Kemasan: Harus tercantum secara eksplisit pada gambar (contoh: "350Ml", "325mL", "600 ml", "1.5L").

### KRITERIA "tidak dikenali":
Setiap kondisi berikut WAJIB menghasilkan `nama_produk`: "tidak dikenali":
- Merek atau jenis produk tidak dapat dipastikan secara jelas.
- Ukuran/volume kemasan (ml/L) buram, terpotong, tertutup, atau tidak terlihat pada gambar.

### ATURAN FORMAT FIELD (NUTRISCAN STANDARD):
1. `nama_produk`:
   - Gunakan nama lengkap produk sesuai yang tertera pada kemasan, diikuti dengan ukuran netto.
   - Penulisan netto disesuaikan dengan teks di kemasan (contoh: "Pocari Sweat Minuman Isotonik 2000Ml", "Aqua Botol 600 ml", "Frestea Minuman Teh Apel 1.5L").
   - Tulis "tidak dikenali" jika Merek ATAU Ukuran/Netto tidak dapat dipastikan.
2. `merek`:
   - Merek utama produk (contoh: "Pocari Sweat", "Adem Sari", "Aqua", "Bear Brand").
3. `jenis`:
   - Isi dengan "Minuman Kemasan" (sesuai kategori default sistem).
4. `kemasan`:
   - Jenis kemasan utama. Pilih salah satu: "Botol", "Kaleng", atau "Kotak".
   - Jika ukuran tidak terbaca/tidak yakin, isi jenis kemasan + ", ukuran tidak terbaca" (contoh: "Botol, ukuran tidak terbaca").
5. `confidence`:
   - Rentang nilai: 0.0 hingga 1.0.
   - Ambil nilai terendah antara keyakinan identitas produk DAN keyakinan keterbacaan netto. Jika netto tidak terbaca, confidence MAKSIMAL adalah 0.4.

---

### FORMAT OUTPUT:
Balas HANYA dengan JSON valid tanpa teks tambahan atau Markdown pembungkus selain standar JSON.

{
  "nama_produk": "...",
  "merek": "...",
  "jenis": "...",
  "kemasan": "...",
  "confidence": 0.0
}
"""

    payload = {
        "contents": [{
            "parts": [
                {"text": prompt},
                {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}}
            ]
        }]
    }

    res = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        json=payload,
        headers={"Content-Type": "application/json"},
        timeout=25
    )
    res.raise_for_status()

    raw = res.json()["candidates"][0]["content"]["parts"][0]["text"]

    # Bersihkan markdown fence jika ada
    clean = raw.replace("```json", "").replace("```", "").strip()

    # Regex fallback: ambil blok JSON pertama dari respons
    match = re.search(r"\{.*?\}", clean, re.DOTALL)
    if match:
        clean = match.group()

    return json.loads(clean)


def identify_product_gemini(image_pil: Image.Image) -> dict:
    """
    Entry point Gemini Vision.
    Kompres gambar → cache by hash → panggil API.
    Return dict dengan key: nama_produk, merek, jenis, confidence [, error].
    """
    try:
        img_bytes, img_hash = _compress_image(image_pil)
        img_b64 = base64.b64encode(img_bytes).decode()
        return _cached_gemini_call(img_hash, img_b64)

    except requests.exceptions.HTTPError as e:
        return {
            "nama_produk": "error", "merek": "", "jenis": "", "confidence": 0.0,
            "error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"
        }
    except Exception as e:
        return {
            "nama_produk": "error", "merek": "", "jenis": "", "confidence": 0.0,
            "error": str(e)
        }


def match_gemini_result_to_db(
    query: str,
    embedder=None,
    vectors=None,
    csv_names=None,
    csv_df=None
) -> tuple:
    """
    Match hasil Gemini ke database dengan SMART PRIORITY MATCHING.
    Return: (match_name, gula, total_gula, kalori) atau (None, 0, 0, 0) jika tidak ditemukan.
    
    Priority:
    1. Search di vector DB (embedding similarity) - PALING AKURAT & RECOMMENDED
       - Menggunakan semantic similarity dengan minimum threshold 0.6
    2. Fallback ke SUGAR_DB string matching (untuk data hardcoded lama)
    3. Return None jika tidak ada match
    """
    if not query or query in ("tidak dikenali", "error"):
        return None, 0, 0, 0
    
    # ── Priority 1: Vector DB Search (Semantic Similarity) ──
    # INI ADALAH METODE UTAMA YANG PALING AKURAT
    if embedder is not None and vectors is not None and csv_names and not csv_df.empty:
        try:
            similar = find_similar(query, embedder, vectors, csv_names, csv_df, top_k=5)
            if similar and len(similar) > 0:
                top_match = similar[0]
                similarity_score = top_match.get("similarity", 0) / 100.0  # Convert % to 0-1
                
                # Debug output untuk troubleshooting
                print(f"[RAG] Query: '{query}'")
                print(f"[RAG] Top match: '{top_match['name']}' (similarity: {top_match.get('similarity', 0)}%)")
                
                # Strict matching: hanya accept jika similarity > 60%
                # Kurangi threshold jika hasil terlalu ketat
                if similarity_score >= 0.6:
                    return (
                        top_match["name"],
                        top_match["gula_per_100ml"],
                        top_match["total_gula"],
                        top_match["kalori"]
                    )
                else:
                    print(f"[RAG] Similarity {similarity_score*100:.1f}% < 60% threshold, fallback to SUGAR_DB")
        except Exception as e:
            print(f"[Warning] Vector DB search error: {e}")
    
    # ── Priority 2: SUGAR_DB String Matching (Simple Fallback) ──
    # Untuk compatibility dengan data hardcoded lama
    q_lower = query.lower()
    for key in SUGAR_DB:
        key_lower = key.lower()
        # More flexible string matching untuk fallback
        if key_lower in q_lower or q_lower in key_lower:
            print(f"[RAG] Fallback to SUGAR_DB: matched '{key}' in query '{query}'")
            gula = SUGAR_DB[key]
            meta = PRODUCT_META.get(key, {"netto": 250, "kalori": 0})
            total_gula = round(gula * meta["netto"] / 100, 1)
            kalori = meta["kalori"]
            return key, gula, total_gula, kalori
    
    # ── Priority 3: Not Found ──
    print(f"[RAG] No match found for query: '{query}'")
    return None, 0, 0, 0

# ===============================
# LOAD YOLO
# ===============================
@st.cache_resource
def load_yolo():
    if os.path.exists(MODEL_PATH):
        return YOLO(MODEL_PATH)
    return None

# ===============================
# UI — GRAD MAP & HELPERS
# ===============================
GRAD_MAP = {
    "A": "linear-gradient(135deg, #005c2e 0%, #00A651 60%, #4CAF50 100%)",
    "B": "linear-gradient(135deg, #2d5a00 0%, #5a8a00 55%, #8BC53F 100%)",
    "C": "linear-gradient(135deg, #7a3800 0%, #c97000 55%, #F9A01B 100%)",
    "D": "linear-gradient(135deg, #7a0007 0%, #c0111a 55%, #ED1C24 100%)",
}

def render_product_card(prod: dict, source: str = "Database", rank: int = 0):
    cfg            = prod["level_cfg"]
    icon_map       = {0: "🥇", 1: "🥈", 2: "🥉"}
    icon           = icon_map.get(rank, "📦")
    gula_per_100ml = prod.get("gula_per_100ml", prod.get("gula", 0))

    st.markdown(f"""
    <div class="ns-pcard" style="background:{GRAD_MAP[prod['level_kode']]};">
        <div class="ns-pcard-header">
            <div>
                <span style="font-size:1.4em;vertical-align:middle;">{icon}</span>
                <span class="ns-pcard-name" style="vertical-align:middle;margin-left:8px;">{prod['name']}</span>
            </div>
            <div class="ns-pcard-lvbadge">{cfg['emoji']} Level {cfg['label']} — {cfg['kategori']}</div>
        </div>
        <div class="ns-pcard-sub">{prod.get('jenis','-')} &nbsp;·&nbsp; Sumber: {source} &nbsp;·&nbsp; {cfg['range']}</div>
        <div class="ns-pcard-grid">
            <div class="ns-pcard-stat">
                <div class="ns-pcard-statval">{gula_per_100ml}</div>
                <div class="ns-pcard-statlbl">g gula/100ml</div>
            </div>
            <div class="ns-pcard-stat">
                <div class="ns-pcard-statval">{prod['total_gula']}</div>
                <div class="ns-pcard-statlbl">g total gula</div>
            </div>
            <div class="ns-pcard-stat">
                <div class="ns-pcard-statval">{prod['kalori']}</div>
                <div class="ns-pcard-statlbl">kkal</div>
            </div>
            <div class="ns-pcard-stat">
                <div class="ns-pcard-statval">{prod['akg']}%</div>
                <div class="ns-pcard-statlbl">% AKG harian</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(min(prod["akg"], 100) / 100, text=f"📊 {prod['akg']}% dari batas AKG harian (50g/hari)")
    if prod["level_kode"] in ("C", "D"):
        st.warning(f"⚠️ {cfg['deskripsi']}  \n**{cfg['rekomendasi']}**")
    else:
        st.success(f"✅ {cfg['deskripsi']}  \n**{cfg['rekomendasi']}**")


def render_comparison_chart(detected_name: str, comp_df: pd.DataFrame):
    st.markdown('<div class="ns-section-title">📊 Perbandingan Semua Produk Database</div>', unsafe_allow_html=True)
    st.caption(
        f"Gabungan YOLO-DB & CSV-DB. "
        f"Produk terdeteksi: **{detected_name}** (disorot ★)"
    )
    if comp_df.empty:
        st.info("CSV database tidak ditemukan, hanya menampilkan YOLO-DB.")

    chart_data = comp_df.copy()
    chart_data["label"] = chart_data["name"].apply(
        lambda n: f"★ {n}" if n == detected_name else n
    )
    st.bar_chart(chart_data.set_index("label")[["gula_per_100ml"]], use_container_width=True)

    # ── ENHANCED TABLE ──────────────────────────────────────────────────
    with st.expander("📋 Lihat Tabel Lengkap", expanded=False):
        LEVEL_COLOR = {"A": "#00A651", "B": "#8BC53F", "C": "#F9A01B", "D": "#ED1C24"}
        LEVEL_BG    = {
            "A": "rgba(0,166,81,0.12)",   "B": "rgba(139,197,63,0.12)",
            "C": "rgba(249,160,27,0.12)", "D": "rgba(237,28,36,0.10)"
        }
        LEVEL_LABEL = {"A": "Lebih Sehat", "B": "Rendah Gula", "C": "Sedang", "D": "Tinggi Gula"}

        gula_max = max(comp_df["gula_per_100ml"].max(), 1)
        target_gula = comp_df.loc[comp_df["name"] == detected_name, "gula_per_100ml"].values
        target_gula_val = float(target_gula[0]) if len(target_gula) > 0 else None

        header_html = """
        <div style="display:grid;grid-template-columns:2fr 1.2fr 1.8fr 0.7fr 0.7fr;
            gap:0;padding:4px 12px 6px;margin-bottom:2px;">
            <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;">Produk</div>
            <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;">Level</div>
            <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;">Gula / 100ml</div>
            <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;text-align:right;">Kalori</div>
            <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;text-align:right;">Sumber</div>
        </div>
        """

        rows_html = ""
        for _, row in comp_df.iterrows():
            lv = row["level"]
            lv_color = LEVEL_COLOR.get(lv, "#888")
            lv_bg    = LEVEL_BG.get(lv, "rgba(128,128,128,0.1)")
            is_detected = row["name"] == detected_name
            gula_val = float(row["gula_per_100ml"])
            bar_pct  = round(gula_val / gula_max * 100, 1)

            if target_gula_val is not None and not is_detected:
                if gula_val < target_gula_val - 0.01:
                    tag = '<span class="ns-winner">✓ Lebih Sehat</span>'
                elif gula_val > target_gula_val + 0.01:
                    tag = '<span class="ns-loser">✗ Lebih Tinggi</span>'
                else:
                    tag = '<span class="ns-same">Sama</span>'
            elif is_detected:
                tag = '<span class="ns-same">★ Terdeteksi</span>'
            else:
                tag = ""

            border_left = "3px solid #00A651" if is_detected else "1px solid rgba(128,128,128,0.1)"
            bg = "rgba(0,166,81,0.04)" if is_detected else "rgba(128,128,128,0.025)"

            rows_html += f"""
            <div style="display:grid;grid-template-columns:2fr 1.2fr 1.8fr 0.7fr 0.7fr;
                gap:0;align-items:center;
                background:{bg};
                border:1px solid rgba(128,128,128,0.09);
                border-left:{border_left};
                border-radius:10px;margin-bottom:5px;padding:10px 12px;">
                <div>
                    <span style="font-weight:700;font-size:13px;">{row['name']}</span>
                    <span style="margin-left:5px;">{tag}</span>
                </div>
                <div>
                    <span style="display:inline-block;font-size:11px;font-weight:700;
                        padding:3px 9px;border-radius:99px;
                        background:{lv_bg};color:{lv_color};
                        border:1.5px solid {lv_color}66;">
                        {lv} — {LEVEL_LABEL.get(lv, lv)}
                    </span>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="flex:1;height:7px;border-radius:99px;background:rgba(128,128,128,0.12);overflow:hidden;">
                        <div style="height:100%;width:{bar_pct}%;background:{lv_color};border-radius:99px;"></div>
                    </div>
                    <span style="font-weight:700;font-size:12px;color:{lv_color};white-space:nowrap;">{gula_val:.1f}g</span>
                </div>
                <div style="text-align:right;font-size:12px;font-weight:600;">{row['kalori']:.0f} kkal</div>
                <div style="text-align:right;font-size:11.5px;opacity:.55;">{row['source']}</div>
            </div>
            """

        st.markdown(f'<div style="margin:8px 0 16px;">{header_html}{rows_html}</div>', unsafe_allow_html=True)

    best  = comp_df.loc[comp_df["gula_per_100ml"].idxmin()]
    worst = comp_df.loc[comp_df["gula_per_100ml"].idxmax()]
    st.markdown(f"""
    <div class="ns-vs-wrap">
        <div class="ns-vs-card ns-vs-best">
            <div class="ns-vs-label">💚 Paling Rendah Gula</div>
            <div class="ns-vs-name">{best['name']}</div>
            <div class="ns-vs-meta">{best['gula_per_100ml']}g / 100ml &nbsp;·&nbsp; Level {best['level']}</div>
        </div>
        <div class="ns-vs-card ns-vs-worst">
            <div class="ns-vs-label">🔴 Paling Tinggi Gula</div>
            <div class="ns-vs-name">{worst['name']}</div>
            <div class="ns-vs-meta">{worst['gula_per_100ml']}g / 100ml &nbsp;·&nbsp; Level {worst['level']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)





# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NutriScan — Deteksi Gula Minuman",
    page_icon="🥤",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — satu kali saja di atas
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=Space+Grotesk:wght@500;700&display=swap');

/* ══════════════════════════════════════════
   GLOBAL RESET & BASE
══════════════════════════════════════════ */
*, *::before, *::after { box-sizing: border-box; }

/* ── Topbar spectrum ── */
.ns-topbar {
    height: 5px;
    background: linear-gradient(90deg,
        #00A651 0%, #4CAF50 20%, #8BC53F 35%,
        #F9A01B 55%, #FF6B35 70%, #ED1C24 100%);
    border-radius: 20px 20px 0 0;
    margin-bottom: 0;
    box-shadow: 0 2px 20px rgba(0,166,81,0.25);
}

/* ── Hero header ── */
.ns-hero {
    padding: 28px 32px 22px;
    border: 1px solid rgba(128,128,128,0.12);
    border-top: none;
    border-radius: 0 0 24px 24px;
    margin-bottom: 28px;
    background: linear-gradient(160deg, rgba(0,166,81,0.04) 0%, transparent 60%);
    backdrop-filter: blur(10px);
}
.ns-brand { display: flex; align-items: center; gap: 16px; margin-bottom: 16px; }
.ns-logo {
    width: 52px; height: 52px; border-radius: 16px; font-size: 28px;
    background: linear-gradient(135deg, rgba(0,166,81,0.15), rgba(139,197,63,0.1));
    border: 1px solid rgba(0,166,81,0.25);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    box-shadow: 0 4px 12px rgba(0,166,81,0.15);
}
.ns-appname {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 26px; font-weight: 700; letter-spacing: -0.6px;
    background: linear-gradient(135deg, #00A651, #4CAF50);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.ns-appsub  { font-size: 13px; opacity: .55; margin-top: 4px; font-family: 'Inter', sans-serif; }
.ns-badges  { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 4px; }
.ns-badge {
    display: inline-flex; align-items: center; gap: 6px;
    font-size: 12px; font-weight: 600; font-family: 'Inter', sans-serif;
    padding: 5px 12px; border-radius: 99px; border: 1.5px solid;
    letter-spacing: 0.2px;
}
.ns-ba { background:rgba(0,166,81,0.08);  color:#00A651; border-color:rgba(0,166,81,0.3); }
.ns-bb { background:rgba(139,197,63,0.08); color:#5a8a00; border-color:rgba(139,197,63,0.35); }
.ns-bc { background:rgba(249,160,27,0.08); color:#c97000; border-color:rgba(249,160,27,0.35); }
.ns-bd { background:rgba(237,28,36,0.07);  color:#c0111a; border-color:rgba(237,28,36,0.3); }
.ns-legal { font-size: 11px; opacity: .35; margin-top: 12px; font-family: 'Inter'; }

/* ── Section titles ── */
.ns-section-title {
    font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 15px; font-weight: 700;
    margin: 22px 0 12px; letter-spacing: -0.1px;
    display: flex; align-items: center; gap: 8px;
}
.ns-section-title::after {
    content: ''; flex: 1; height: 1px;
    background: linear-gradient(90deg, rgba(128,128,128,0.15), transparent);
}

/* ── Nutri-level grid ── */
.ns-lv-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0,1fr));
    gap: 10px;
    padding: 4px 0 10px;
}
.ns-lv-card {
    border-radius: 16px;
    padding: 14px 13px;
    border: 1px solid rgba(128,128,128,0.1);
    background: rgba(128,128,128,0.025);
    transition: transform .15s;
}
.ns-lv-card:hover { transform: translateY(-1px); }
.ns-lv-dot {
    width: 32px; height: 32px; border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 14px; font-weight: 800; color: white;
    margin-bottom: 9px;
    font-family: 'Space Grotesk', sans-serif;
    box-shadow: 0 3px 10px rgba(0,0,0,0.18);
}
.ns-lv-name  { font-size: 13px; font-weight: 700; margin-bottom: 4px; }
.ns-lv-range {
    font-size: 10px; font-family: 'Space Grotesk', monospace;
    padding: 2px 8px; border-radius: 99px;
    display: inline-block; border: 1px solid; margin-bottom: 7px;
    font-weight: 600;
}
.ns-lv-desc  { font-size: 11.5px; opacity: .5; line-height: 1.55; }

/* ══════════════════════════════════════════
   COMPARISON TABLE — Premium Style
══════════════════════════════════════════ */
.ns-comp-wrap { overflow-x: auto; margin: 8px 0 20px; }
.ns-comp-table {
    width: 100%; border-collapse: separate; border-spacing: 0 6px;
    font-family: 'Inter', sans-serif; font-size: 13px;
}
.ns-comp-table thead th {
    font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.9px; opacity: .45; padding: 0 12px 8px; text-align: left;
}
.ns-comp-table thead th.right { text-align: right; }
.ns-comp-row {
    transition: transform .12s;
}
.ns-comp-row:hover .ns-comp-cell { opacity: .92; }
.ns-comp-cell {
    padding: 12px 12px; background: rgba(128,128,128,0.04);
    border-top: 1px solid rgba(128,128,128,0.07);
    border-bottom: 1px solid rgba(128,128,128,0.07);
}
.ns-comp-cell:first-child {
    border-left: 1px solid rgba(128,128,128,0.07);
    border-radius: 12px 0 0 12px;
}
.ns-comp-cell:last-child {
    border-right: 1px solid rgba(128,128,128,0.07);
    border-radius: 0 12px 12px 0;
}
.ns-comp-cell.right { text-align: right; }
.ns-comp-target .ns-comp-cell {
    background: rgba(0,166,81,0.06) !important;
    border-color: rgba(0,166,81,0.18) !important;
}
.ns-comp-target .ns-comp-cell:first-child {
    border-left-color: #00A651 !important; border-left-width: 3px !important;
}

/* Level badge pill */
.ns-lvbadge {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; font-weight: 700;
    padding: 3px 9px; border-radius: 99px;
    white-space: nowrap;
}

/* Sugar bar */
.ns-sugarbar-wrap {
    display: flex; align-items: center; gap: 8px;
    min-width: 120px;
}
.ns-sugarbar-track {
    flex: 1; height: 7px; border-radius: 99px;
    background: rgba(128,128,128,0.12); overflow: hidden;
}
.ns-sugarbar-fill {
    height: 100%; border-radius: 99px;
    transition: width .3s;
}
.ns-sugarval { font-weight: 700; font-size: 12px; white-space: nowrap; }

/* Winner / loser badges */
.ns-winner { display: inline-block; font-size: 10px; font-weight: 700;
    background: linear-gradient(135deg, #00A651, #4CAF50); color: white;
    padding: 2px 7px; border-radius: 99px; margin-left: 5px; letter-spacing: 0.3px; }
.ns-loser  { display: inline-block; font-size: 10px; font-weight: 700;
    background: linear-gradient(135deg, #ED1C24, #c0111a); color: white;
    padding: 2px 7px; border-radius: 99px; margin-left: 5px; letter-spacing: 0.3px; }
.ns-same   { display: inline-block; font-size: 10px; font-weight: 700;
    background: rgba(128,128,128,0.15); color: rgba(128,128,128,0.8);
    padding: 2px 7px; border-radius: 99px; margin-left: 5px; letter-spacing: 0.3px; }

/* ══════════════════════════════════════════
   COMPARISON SUMMARY CARDS
══════════════════════════════════════════ */
.ns-vs-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin: 16px 0; }
.ns-vs-card {
    border-radius: 16px; padding: 16px;
    border: 1px solid rgba(128,128,128,0.1);
}
.ns-vs-best {
    background: linear-gradient(135deg, rgba(0,166,81,0.1), rgba(139,197,63,0.07));
    border-color: rgba(0,166,81,0.2);
}
.ns-vs-worst {
    background: linear-gradient(135deg, rgba(237,28,36,0.08), rgba(237,28,36,0.04));
    border-color: rgba(237,28,36,0.18);
}
.ns-vs-label { font-size: 10px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.9px; opacity: .5; margin-bottom: 8px; }
.ns-vs-name { font-size: 15px; font-weight: 700; margin-bottom: 4px; }
.ns-vs-meta { font-size: 12px; opacity: .6; }

/* ══════════════════════════════════════════
   PRODUCT CARD — Cinematic
══════════════════════════════════════════ */
.ns-pcard {
    border-radius: 20px; padding: 22px 24px; margin-bottom: 16px;
    color: white; position: relative; overflow: hidden;
    box-shadow: 0 8px 32px rgba(0,0,0,0.25);
}
.ns-pcard::before {
    content: ''; position: absolute; top: -40%; right: -10%;
    width: 200px; height: 200px; border-radius: 50%;
    background: rgba(255,255,255,0.06); pointer-events: none;
}
.ns-pcard-header { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px; }
.ns-pcard-name { font-family: 'Space Grotesk', 'Inter', sans-serif;
    font-size: 1.25em; font-weight: 700; }
.ns-pcard-lvbadge {
    background: rgba(255,255,255,0.18); padding: 4px 13px; border-radius: 99px;
    font-size: 0.8em; font-weight: 700; backdrop-filter: blur(6px);
    border: 1px solid rgba(255,255,255,0.25);
    white-space: nowrap;
}
.ns-pcard-sub { font-size: 0.82em; opacity: 0.75; margin-bottom: 16px; }
.ns-pcard-grid {
    display: grid; grid-template-columns: repeat(4,1fr); gap: 8px;
    text-align: center; margin: 12px 0;
}
.ns-pcard-stat {
    background: rgba(0,0,0,0.18); padding: 10px 6px; border-radius: 12px;
    backdrop-filter: blur(4px);
}
.ns-pcard-statval { font-size: 1.3em; font-weight: 800; line-height: 1; margin-bottom: 4px; }
.ns-pcard-statlbl { font-size: 0.68em; opacity: 0.75; }
.ns-pcard-range { font-size: 0.75em; opacity: 0.7; margin-top: 8px;
    font-family: 'Space Grotesk', monospace; }

/* ══════════════════════════════════════════
   SUMMARY PANEL (top-right)
══════════════════════════════════════════ */
.ns-summary {
    border-radius: 18px; padding: 18px 20px;
    border: 1px solid rgba(128,128,128,0.1);
    background: rgba(128,128,128,0.025);
    margin-bottom: 16px;
}
.ns-summary-title { font-size: 10.5px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.8px; opacity: .4; margin-bottom: 14px; }
.ns-summary-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
.ns-summary-item strong { font-size: 11px; opacity: .5; display: block;
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 4px; }
.ns-summary-item span { font-size: 14px; font-weight: 700; }

/* ── Buttons ── */
.stButton > button {
    border-radius: 12px !important;
    font-weight: 700 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
    padding: 10px 0 !important;
    min-height: 46px !important;
    letter-spacing: 0.1px !important;
}

/* ── Metric text bigger ── */
[data-testid="stMetricValue"] { font-size: 22px !important; font-weight: 800 !important; font-family: 'Space Grotesk', sans-serif !important; }
[data-testid="stMetricLabel"] { font-size: 12px !important; font-weight: 500 !important; opacity: 0.6 !important; }

/* ── Progress bar color override ── */
[data-testid="stProgress"] > div > div { border-radius: 99px !important; }

/* ── Tabs ── */
[data-testid="stTabs"] [role="tab"] { font-weight: 600 !important; font-size: 13.5px !important; }
</style>
""", unsafe_allow_html=True)


# ── EXPANDER: PANDUAN NUTRI-LEVEL (dibangun di sini, dipakai di bawah) ─────────
_LEVEL_DOT = {"A": "#2E7D32", "B": "#8BC34A", "C": "#F57C00", "D": "#C62828"}
_LEVEL_BG  = {"A": "#E8F5E9", "B": "#F1F8E9", "C": "#FFF3E0", "D": "#FFEBEE"}
_LEVEL_CLR = {"A": "#1B5E20", "B": "#33691E", "C": "#E65100", "D": "#B71C1C"}
_LEVEL_BDR = {"A": "#2E7D32", "B": "#8BC34A", "C": "#F57C00", "D": "#C62828"}

_cards = ""
for _kode, _cfg in NUTRI_LEVEL_CONFIG.items():
    _cards += f"""
    <div class="ns-lv-card">
        <div class="ns-lv-dot" style="background:{_LEVEL_DOT[_kode]}">{_kode}</div>
        <div class="ns-lv-name">{_cfg['kategori']}</div>
        <div class="ns-lv-range" style="color:{_LEVEL_CLR[_kode]};background:{_LEVEL_BG[_kode]};border-color:{_LEVEL_BDR[_kode]}">{_cfg['range']}</div>
        <div class="ns-lv-desc">{_cfg['deskripsi']}</div>
    </div>"""

# ── TOP BAR + HERO ────────────────────────────────────────────────────────────
st.markdown('<div class="ns-topbar"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="ns-hero">
    <div class="ns-brand">
        <div class="ns-logo">🥤</div>
        <div>
            <div class="ns-appname">NutriScan</div>
            <div class="ns-appsub">Sistem Deteksi Kadar Gula &amp; Klasifikasi Nutri-Level Minuman Kemasan Indonesia</div>
        </div>
    </div>
<div class="ns-legal">
        KMK No. HK.01.07/MENKES/301/2026 &nbsp;·&nbsp; AKG Gula: 50g/hari (WHO &amp; Permenkes No.30/2013) &nbsp;·&nbsp; Powered by YOLOv8 &amp; Gemini Vision
</div>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="ns-section-title">📘 Panduan Nutri-Level</div>', unsafe_allow_html=True)
st.markdown(f'<div class="ns-lv-grid">{_cards}</div>', unsafe_allow_html=True)








# ══════════════════════════════════════════════════════════════════════════════
# Load Mode;
# ══════════════════════════════════════════════════════════════════════════════

# ── LOAD RESOURCES ────────────────────────────────────────────────────────────
yolo_model = load_yolo()
csv_df     = load_csv_db()
comp_df    = build_comparison_db(csv_df)

embedder, vectors, csv_names = build_vector_db(csv_df)
has_vector_db = embedder is not None



# ══════════════════════════════════════════════════════════════════════════════
# Kontrol Inline — nilai default dan migrasi state
# ══════════════════════════════════════════════════════════════════════════════
if "yolo_conf" not in st.session_state:
    st.session_state.yolo_conf = st.session_state.get("sidebar_yolo_conf", YOLO_CONF_THRESHOLD)
if "stabilize_n" not in st.session_state:
    st.session_state.stabilize_n = st.session_state.get("sidebar_stabilize", STABILIZE_FRAMES)
if "freeze_delay" not in st.session_state:
    st.session_state.freeze_delay = st.session_state.get("sidebar_freeze_delay", 0.5)

# ── TOMBOL REFRESH DATA ────────────────────────────────────────────────────────
col_refresh_left, col_refresh_mid, col_refresh_right = st.columns([1, 6, 1])
with col_refresh_left:
    pass  # spacer
with col_refresh_mid:
    st.markdown(
        """
        <div style='font-size:12px; color:rgba(255,255,255,0.6); text-align:center;'>
        <em>💡 Jika menambah data di Admin Panel, klik <strong>Refresh Data</strong> →</em>
        </div>
        """, unsafe_allow_html=True
    )
with col_refresh_right:
    if st.button("🔄 Refresh Data", use_container_width=True, help="Clear cache & muat data terbaru dari database"):
        if hasattr(st.cache_data, "clear"):
            st.cache_data.clear()
        st.rerun()
 





# ══════════════════════════════════════════════════════════════════════════════
# PILIH METODE DETEKSI
# ══════════════════════════════════════════════════════════════════════════════
UPLOAD_MODE = "📤 Upload File"
CAMERA_MODE = "🎥 Real-time Kamera"
MANUAL_MODE = "📝 Pilih Manual"

if "selected_mode" not in st.session_state:
    st.session_state.selected_mode = UPLOAD_MODE

st.markdown('<div class="ns-section-title">Pilih Cara Deteksi</div>', unsafe_allow_html=True)

cols = st.columns(3, gap='medium')
with cols[0]:
    if st.button("📤  Upload Foto", key="mode_tab_upload", use_container_width=True,
                 type="primary" if st.session_state.selected_mode == UPLOAD_MODE else "secondary"):
        st.session_state.selected_mode = UPLOAD_MODE
        st.rerun()
with cols[1]:
    if st.button("🎥  Kamera Langsung", key="mode_tab_camera", use_container_width=True,
                 type="primary" if st.session_state.selected_mode == CAMERA_MODE else "secondary"):
        st.session_state.selected_mode = CAMERA_MODE
        st.rerun()
with cols[2]:
    if st.button("📝  Pilih Manual", key="mode_tab_manual", use_container_width=True,
                 type="primary" if st.session_state.selected_mode == MANUAL_MODE else "secondary"):
        st.session_state.selected_mode = MANUAL_MODE
        st.rerun()

mode = st.session_state.selected_mode
st.markdown("<div style='margin-bottom:20px'></div>", unsafe_allow_html=True)

yolo_thresh = st.session_state.yolo_conf
stabilize_n = st.session_state.stabilize_n
freeze_delay = st.session_state.freeze_delay


# ════════════════════════════════════════════════════════════════════
# MODE 1 — UPLOAD / FOTO
# ════════════════════════════════════════════════════════════════════
if mode == UPLOAD_MODE:
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="ns-section-title">📸 Foto Produk</div>', unsafe_allow_html=True)
        # Tabs upload / kamera
        tab_upload, tab_camera = st.tabs(["📁 Upload File", "📷 Ambil Foto (Kamera)"])
        image = None
        input_source = None

        with tab_upload:
            uploaded = st.file_uploader(
                "Pilih foto minuman kemasan", type=["jpg","jpeg","png"],
                help="Foto tampak depan produk dengan label yang terlihat jelas."
            )
            if uploaded:
                image = Image.open(uploaded).convert("RGB")
                input_source = "Upload"
                st.image(image, caption="Gambar yang diupload", use_container_width=True)

        with tab_camera:
            cam_img = st.camera_input(
                "Arahkan kamera ke produk minuman",
                help="Pastikan label terlihat jelas."
            )
            if cam_img:
                image = Image.open(cam_img).convert("RGB")
                input_source = "Kamera"
                st.image(image, caption="Foto dari kamera", use_container_width=True)

        if image:
            st.markdown("---")
            use_gemini = st.checkbox("🤖 Gunakan Gemini Vision (Lebih Akurat)", value=True)
            btn_analyze = st.button("🔍  Analisis Sekarang", type="primary", use_container_width=True)

    with col_right:
        st.markdown('<div class="ns-section-title">📊 Hasil Analisis</div>', unsafe_allow_html=True)
        summary_slot = st.container()
        render_summary_panel(None)

        if image and btn_analyze:
            found_product = None

            if use_gemini:
                with st.spinner("Sedang menganalisis gambar..."):
                    hasil_gemini = identify_product_gemini(image)

                if hasil_gemini.get("nama_produk") in ["tidak dikenali", "error"]:
                    st.warning(f"⚠️ Gemini tidak bisa mengenali produk secara otomatis. Error: {hasil_gemini.get('error','')}")
                    gemini_failed = True
                else:
                    gemini_failed = False
            else:
                st.info("ℹ️ Gemini dinonaktifkan. Silakan pilih produk secara manual dari database.")
                gemini_failed = True

            if gemini_failed:
                manual_options = csv_names if csv_names else list(SUGAR_DB.keys())
                st.markdown("#### 📋 Pilih Produk Manual")
                manual_pick = st.selectbox(
                    "Pilih produk yang paling sesuai dari database:",
                    manual_options,
                    key="manual_pick_upload"
                )
                if st.button("✅ Konfirmasi Pilihan Manual"):
                    # Persist manual choice across rerun so we can render analysis
                    st.session_state["manual_pick_upload"] = manual_pick
                    st.session_state["manual_pick_upload_confirm"] = True
                    st.rerun()
            else:
                # Jika Gemini berhasil, lakukan pencarian di Vector DB (Ciri khas app2)[cite: 1]
                nama = hasil_gemini["nama_produk"]
                merek = hasil_gemini.get("merek", "")
                st.success(f"**Gemini mengenali:** {merek} {nama}")

                query = f"{merek} {nama}".strip()
                if has_vector_db and embedder is not None and vectors is not None:
                    similar = find_similar(query, embedder, vectors, csv_names, csv_df, top_k=5)
                    main_res = similar[0]
                    match_confidence = main_res.get("similarity", 0)
                    
                    # Tampilkan confidence matching ke user
                    if match_confidence >= 75:
                        st.info(f"✅ Cocok dengan database: **{main_res['name']}** ({match_confidence}% kesamaan)")
                    elif match_confidence >= 60:
                        st.info(f"⚠️ Agak cocok dengan: **{main_res['name']}** ({match_confidence}% kesamaan). Verifikasi manual disarankan.")
                    else:
                        st.warning(f"❓ Cocok lemah dengan: **{main_res['name']}** ({match_confidence}% kesamaan). Silakan pilih manual.")
                    
                    level_kode, level_cfg = get_nutri_level(main_res["gula_per_100ml"])
                    found_product = {
                        "name": main_res["name"],
                        "gula": main_res["gula_per_100ml"],
                        "total_gula": main_res["total_gula"],
                        "akg": round((main_res["total_gula"] / AKG_GULA) * 100, 1),
                        "kategori": level_cfg["label"],
                        "level_kode": level_kode,
                        "level_cfg": level_cfg,
                        "kalori": main_res["kalori"],
                        "confidence": hasil_gemini.get("confidence", 0.85),
                        "source": "Gemini",
                        "all_similar": similar,
                        "matching_confidence": match_confidence
                    }
                else:
                    info = get_product_info(nama)
                    found_product = {
                        "name": info["name"],
                        "gula": info["gula_per_100ml"],
                        "total_gula": info["total_gula"],
                        "akg": info["akg"],
                        "kategori": info["level_kode"],
                        "level_kode": info["level_kode"],
                        "level_cfg": info["level_cfg"],
                        "kalori": info["kalori"],
                        "confidence": hasil_gemini.get("confidence", 0.85),
                        "source": "Gemini"
                    }

            # ── Render Hasil (UI) ──
            if found_product:
                summary_slot.empty()
                render_summary_panel(found_product)
                render_upload_analysis(found_product, found_product.get("all_similar"))

            # Handle manual selection persisted across reruns (upload flow)
            if st.session_state.get("manual_pick_upload_confirm"):
                manual_choice = st.session_state.get("manual_pick_upload")
                if manual_choice:
                    found_product = build_manual_product_result(manual_choice, csv_df=csv_df)
                    if has_vector_db and embedder is not None and vectors is not None:
                        found_product["all_similar"] = find_similar(manual_choice, embedder, vectors, csv_names, csv_df, top_k=5)
                    summary_slot.empty()
                    render_summary_panel(found_product)
                    render_upload_analysis(found_product, found_product.get("all_similar"))
                # clear the flag so it doesn't re-trigger repeatedly
                st.session_state["manual_pick_upload_confirm"] = False

        elif not image:
            st.info("💡 Silakan upload foto atau ambil gambar terlebih dahulu di panel kiri.")

# ════════════════════════════════════════════════════════════════════
# MODE 2 — KAMERA REAL-TIME
# ════════════════════════════════════════════════════════════════════
elif mode == CAMERA_MODE:
    st.markdown('<div class="ns-section-title">🎥 Kamera Real-time — Deteksi Otomatis</div>', unsafe_allow_html=True)

    if yolo_model is None:
        st.error(
            f"❌ Model `{MODEL_PATH}` tidak ditemukan. "
            "Gunakan mode Upload sebagai alternatif."
        )
        st.stop()

    with st.expander("ℹ️ Cara Kerja Kamera Real-time", expanded=False):
        st.markdown(f"""
| Kondisi | Aksi |
|---|---|
| 🎯 Produk terdeteksi stabil | Frame **freeze otomatis** + tampil hasil |
| ❓ Deteksi kurang yakin | Frame disimpan → **tombol "Kirim ke Gemini"** muncul |
| ✅ Setelah freeze | Tersedia **tombol "Koreksi via Gemini"** jika hasil kurang tepat |
| 🚫 Tidak ada produk | Kamera terus scan otomatis |
| 🔄 Scan Ulang | Reset semua, mulai dari awal |
        """)

    # ── Session state ─────────────────────────────────────────────────────────
    for key, default in {
        "rt_frozen":            False,
        "rt_final_det":         {},
        "rt_frozen_frame":      None,   # frame setelah freeze (YOLO confident)
        "rt_pending_frame":     None,   # frame saat low-conf
        "rt_pending_boxes":     [],
        "rt_show_gemini_btn":   False,  # tombol Gemini untuk low-conf
        "rt_show_gemini_corr":  False,  # tombol Gemini koreksi untuk confident
        "rt_freeze_timer":      None,
        "rt_lowconf_timer":     None,   # timer untuk delay tampil panel Gemini (low-conf)
        "rt_nodetect_counter":  0,      # counter frame tanpa deteksi
        "rt_capture_pending":   False,  # flag untuk capture via tombol
        "rt_capture_frame":     None,   # frame yang di-capture
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ── Layout ────────────────────────────────────────────────────────────────
    col_cam, col_info = st.columns([1.3, 1], gap="large")

    with col_cam:
        btn_cols = st.columns([1, 1, 1])
        with btn_cols[0]:
            run = st.checkbox("▶️ Mulai Kamera", key="cam_run")
        with btn_cols[1]:
            reset_btn = st.button("🔄 Scan Ulang", help="Reset semua dan mulai dari awal")
        with btn_cols[2]:
            capture_btn = st.button("📸 Capture", help="Tangkap frame saat ini dan kirim ke Gemini")

        frame_window = st.image([])
        status_box   = st.empty()
        gemini_area  = st.empty()   # Panel Gemini (low-conf atau koreksi)

     
    with col_info:
        st.markdown('<div class="ns-section-title">📦 Hasil Deteksi</div>', unsafe_allow_html=True)
        hasil_placeholder   = st.empty()
        chart_placeholder   = st.empty()
        compare_placeholder = st.empty()

    # ── Reset ─────────────────────────────────────────────────────────────────
    if reset_btn:
        for key in ["rt_frozen","rt_final_det","rt_frozen_frame","rt_pending_frame",
                    "rt_pending_boxes","rt_show_gemini_btn","rt_show_gemini_corr","rt_freeze_timer",
                    "rt_lowconf_timer","rt_nodetect_counter","rt_capture_pending","rt_capture_frame"]:
            st.session_state[key] = (
                False if key in ("rt_frozen","rt_show_gemini_btn","rt_show_gemini_corr","rt_capture_pending") else
                {} if "det" in key else
                None if "frame" in key or "timer" in key else
                [] if "boxes" in key else 0
            )
        hasil_placeholder.empty()
        chart_placeholder.empty()
        compare_placeholder.empty()
        gemini_area.empty()
        st.rerun()

    # ── Render hasil panel kanan ───────────────────────────────────────────────
    def render_rt_hasil(det_dict: dict):
        if len(det_dict) == 1:
            name, d = next(iter(det_dict.items()))
            if "total_gula" not in d:
                netto = PRODUCT_META.get(name, {}).get("netto", 250)
                d["total_gula"] = round(d.get("gula", 0) * netto / 100, 1)

            found_product = {
                "name": name,
                "gula": d.get("gula", d.get("gula_per_100ml", 0)),
                "total_gula": d["total_gula"],
                "akg": d["akg"],
                "kategori": d["level_cfg"]["label"],
                "level_kode": d["level_kode"],
                "level_cfg": d["level_cfg"],
                "kalori": d["kalori"],
                "source": d.get("source", "YOLO"),
                "all_similar": d.get("all_similar"),
            }
            with hasil_placeholder.container():
                render_upload_analysis(found_product, found_product.get("all_similar"))
            return

        with hasil_placeholder.container():
            for name, d in det_dict.items():
                cfg      = d["level_cfg"]
                src_icon = {"Gemini":"🤖","Manual":"👆"}.get(d["source"], "🎯")
                lv_kode  = d["level_kode"]
                st.markdown(f"""
                <div class="ns-pcard" style="background:{GRAD_MAP[lv_kode]};">
                    <div class="ns-pcard-header">
                        <div>
                            <span style="font-size:1.3em;vertical-align:middle;">{src_icon}</span>
                            <span class="ns-pcard-name" style="vertical-align:middle;margin-left:8px;">{name}</span>
                        </div>
                        <div class="ns-pcard-lvbadge">{cfg['emoji']} Level {cfg['label']} — {cfg['kategori']}</div>
                    </div>
                    <div class="ns-pcard-sub">{cfg['range']} &nbsp;·&nbsp; Sumber: {d['source']}</div>
                    <div class="ns-pcard-grid" style="grid-template-columns:1fr 1fr;">
                        <div class="ns-pcard-stat">
                            <div class="ns-pcard-statval">{d['gula']}g</div>
                            <div class="ns-pcard-statlbl">per 100ml</div>
                        </div>
                        <div class="ns-pcard-stat">
                            <div class="ns-pcard-statval">{d['akg']}%</div>
                            <div class="ns-pcard-statlbl">AKG harian</div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)
                st.progress(min(d["akg"], 100) / 100, text=f"Kontribusi AKG: {d['akg']}%")
                if d["level_kode"] in ("C", "D"):
                    st.warning(cfg["rekomendasi"])
                else:
                    st.success(cfg["rekomendasi"])

            if len(det_dict) >= 2:
                items = sorted(det_dict.items(), key=lambda x: x[1]["gula"])
                bn, bd = items[0]; wn, wd = items[-1]
                LEVEL_COLOR = {"A": "#00A651", "B": "#8BC53F", "C": "#F9A01B", "D": "#ED1C24"}
                gula_max = max(float(wd['gula']), 1)
                rows_html = ""
                for _n, _d in items:
                    lv = _d["level_kode"]
                    lv_color = LEVEL_COLOR.get(lv, "#888")
                    bar_pct = round(float(_d["gula"]) / gula_max * 100, 1)
                    is_best = (_n == bn)
                    tag = '<span class="ns-winner">✓ Lebih Sehat</span>' if is_best else '<span class="ns-loser">✗ Lebih Tinggi</span>'
                    border_left = "3px solid #00A651" if is_best else "1px solid rgba(128,128,128,0.1)"
                    bg = "rgba(0,166,81,0.04)" if is_best else "rgba(128,128,128,0.025)"
                    rows_html += f"""
                    <div style="display:grid;grid-template-columns:2fr 2fr 1fr;
                        gap:0;align-items:center;
                        background:{bg};border:1px solid rgba(128,128,128,0.09);
                        border-left:{border_left};border-radius:10px;
                        margin-bottom:5px;padding:10px 12px;">
                        <div><span style="font-weight:700;font-size:13px;">{_n}</span> {tag}</div>
                        <div style="display:flex;align-items:center;gap:8px;">
                            <div style="flex:1;height:7px;border-radius:99px;background:rgba(128,128,128,0.12);overflow:hidden;">
                                <div style="height:100%;width:{bar_pct}%;background:{lv_color};border-radius:99px;"></div>
                            </div>
                            <span style="font-weight:700;font-size:12px;color:{lv_color};white-space:nowrap;">{_d['gula']}g</span>
                        </div>
                        <div style="text-align:right;font-size:12px;font-weight:600;">{_d['akg']}% AKG</div>
                    </div>"""
                header_mini = """
                <div style="display:grid;grid-template-columns:2fr 2fr 1fr;
                    padding:4px 12px 6px;margin-bottom:2px;">
                    <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;">Produk</div>
                    <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;">Gula / 100ml</div>
                    <div style="font-size:10.5px;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;opacity:.4;text-align:right;">AKG</div>
                </div>"""
                st.markdown(f"""
                <div style="margin:12px 0 8px;">{header_mini}{rows_html}</div>
                <div class="ns-vs-wrap" style="margin-top:10px;">
                    <div class="ns-vs-card ns-vs-best">
                        <div class="ns-vs-label">💚 Lebih Direkomendasikan</div>
                        <div class="ns-vs-name">{bn}</div>
                        <div class="ns-vs-meta">Level {bd['level_kode']} · {bd['gula']}g/100ml</div>
                    </div>
                    <div class="ns-vs-card ns-vs-worst">
                        <div class="ns-vs-label">🔴 Lebih Tinggi Gula</div>
                        <div class="ns-vs-name">{wn}</div>
                        <div class="ns-vs-meta">Level {wd['level_kode']} · {wd['gula']}g/100ml</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        first_name = next(iter(det_dict))
        with compare_placeholder.container():
            render_comparison_chart(first_name, comp_df)

    # ── Tampilkan frozen frame jika sudah ada ──────────────────────────────────
    if st.session_state.rt_frozen:
        if st.session_state.rt_frozen_frame is not None:
            frame_window.image(cv2.cvtColor(st.session_state.rt_frozen_frame, cv2.COLOR_BGR2RGB))
        status_box.success("✅ Deteksi selesai. Tekan 🔄 Scan Ulang untuk mulai lagi.")
        if st.session_state.rt_final_det:
            render_rt_hasil(st.session_state.rt_final_det)
            
            # Reset timer low-conf untuk memastikan clean state
            st.session_state.rt_lowconf_timer = None
            st.session_state.rt_show_gemini_btn = False

        # ── TOMBOL KOREKSI GEMINI (setelah YOLO confident berhasil) ──────────
        # Selalu tampil setelah freeze agar user bisa koreksi jika salah
        with gemini_area.container():
            st.markdown('<div class="gemini-panel-ok">', unsafe_allow_html=True)
            st.info(
                "🤖 **Prediksi YOLO dirasa kurang tepat?**\n\n"
                "Tekan tombol di bawah untuk mengirim frame yang dibekukan ke **Gemini Vision** sebagai koreksi."
            )
            col_gc1, col_gc2 = st.columns(2)
            with col_gc1:
                send_correction = st.button(
                    "🤖 Koreksi via Gemini", type="secondary", use_container_width=True,
                    key="btn_gemini_corr"
                )
            with col_gc2:
                dismiss_corr = st.button(
                    "✅ Hasil Sudah Benar", use_container_width=True,
                    key="btn_dismiss_corr"
                )
            st.markdown('</div>', unsafe_allow_html=True)

            if dismiss_corr:
                gemini_area.empty()

            if send_correction and st.session_state.rt_frozen_frame is not None:
                pil_f = Image.fromarray(
                    cv2.cvtColor(st.session_state.rt_frozen_frame, cv2.COLOR_BGR2RGB)
                )
                with st.spinner("🤖 Gemini sedang menganalisis frame..."):
                    hasil_g = identify_product_gemini(pil_f)

                g_name = hasil_g.get("nama_produk", "")
                
                # ── Proses hasil Gemini (sama seperti UPLOAD_MODE) ───────────────────
                if g_name not in ["tidak dikenali", "error"]:
                    # Gemini berhasil mengenali produk
                    merek = hasil_g.get("merek", "")
                    st.success(f"**Gemini mengenali:** {merek} {g_name}")
                    
                    query = f"{merek} {g_name}".strip()
                    
                    # Search di Vector DB dengan proper matching
                    match, gula, total_gula, kalori = match_gemini_result_to_db(
                        query, embedder, vectors, csv_names, csv_df
                    )
                    
                    # Jika produk ditemukan di database
                    if match:
                        lv_k, lv_cfg = get_nutri_level(gula)
                        akg = round((total_gula / AKG_GULA) * 100, 1)

                        gemini_det = {match: {
                            "gula": gula, "total_gula": total_gula, "akg": akg,
                            "level_kode": lv_k, "level_cfg": lv_cfg,
                            "kalori": kalori, "confidence": hasil_g.get("confidence", 0.85), "source": "Gemini",
                        }}
                        # Gambar ulang bounding box di frozen frame
                        frm = st.session_state.rt_frozen_frame.copy()
                        h, w = frm.shape[:2]
                        color = lv_cfg["cv2_bgr"]
                        cv2.rectangle(frm, (10, 10), (w-10, h-10), color, 3)
                        cv2.putText(frm, f"{match} [{lv_k}] Gemini",
                                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                        cv2.putText(frm, f"{gula}g/100ml | AKG:{akg}%",
                                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

                        st.session_state.rt_frozen_frame = frm
                        st.session_state.rt_final_det    = gemini_det
                        st.success(
                            f"✅ Gemini mengoreksi: **{match}** → {lv_cfg['label']} "
                            f"({lv_cfg['emoji']} {gula}g/100ml)"
                        )
                        st.rerun()
                    else:
                        # Produk tidak ada di database, tawarkan pilih manual
                        conf_g = hasil_g.get("confidence", 0)
                        st.warning(
                            f"⚠️ Gemini mengenali **{g_name}** (conf {conf_g*100:.0f}%) "
                            "tapi tidak ada di database."
                        )
                        manual_corr = st.selectbox(
                            "Pilih produk yang paling sesuai:", list(SUGAR_DB.keys()),
                            key="sel_corr"
                        )
                        if st.button("✅ Gunakan Koreksi Manual", key="btn_corr_manual", use_container_width=True):
                            corr_det = {
                                manual_corr: build_manual_product_result(manual_corr, source="Manual", confidence=0.0, csv_df=csv_df)
                            }
                            st.session_state.rt_final_det = corr_det
                            st.rerun()
                else:
                    st.error(
                        f"❌ Gemini tidak dapat mengenali produk.\n"
                        f"**Detail:** {hasil_g.get('error','–')}"
                    )

    # ── Handle tombol Capture (kirim frame ke Gemini) ─────────────────────────
    if capture_btn:
        st.session_state.rt_capture_pending = True

    if st.session_state.rt_capture_pending and st.session_state.rt_capture_frame is not None:
        with gemini_area.container():
            st.markdown('<div class="gemini-panel">', unsafe_allow_html=True)
            st.info(
                "📸 **Frame di-capture!**\n\n"
                "Mengirim ke Gemini untuk analisis..."
            )
            
            pil_capture = Image.fromarray(
                cv2.cvtColor(st.session_state.rt_capture_frame, cv2.COLOR_BGR2RGB)
            )
            st.image(pil_capture, caption="Frame yang di-capture", width=300)
            
            with st.spinner("🤖 Gemini sedang menganalisis frame..."):
                hasil_g = identify_product_gemini(pil_capture)
            
            g_name = hasil_g.get("nama_produk", "")
            
            # ── Proses hasil Gemini (sama seperti UPLOAD_MODE) ───────────────────
            if g_name not in ["tidak dikenali", "error"]:
                # Gemini berhasil mengenali produk
                merek = hasil_g.get("merek", "")
                st.success(f"**Gemini mengenali:** {merek} {g_name}")
                
                query = f"{merek} {g_name}".strip()
                
                # Search di Vector DB dengan proper matching
                match, gula, total_gula, kalori = match_gemini_result_to_db(
                    query, embedder, vectors, csv_names, csv_df
                )
                
                # Jika produk ditemukan di database
                if match:
                    lv_k, lv_cfg = get_nutri_level(gula)
                    akg = round((total_gula / AKG_GULA) * 100, 1)
                    
                    gemini_det = {match: {
                        "gula": gula, "total_gula": total_gula, "akg": akg,
                        "level_kode": lv_k, "level_cfg": lv_cfg,
                        "kalori": kalori, "confidence": hasil_g.get("confidence", 0.85), "source": "Gemini",
                    }}
                    
                    # Gambar bounding box di frame yang di-capture
                    frm = st.session_state.rt_capture_frame.copy()
                    h, w = frm.shape[:2]
                    color = lv_cfg["cv2_bgr"]
                    cv2.rectangle(frm, (10, 10), (w-10, h-10), color, 3)
                    cv2.putText(frm, f"{match} [{lv_k}] Gemini",
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(frm, f"{gula}g/100ml | AKG:{akg}%",
                                (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                    
                    st.session_state.rt_frozen         = True
                    st.session_state.rt_frozen_frame   = frm
                    st.session_state.rt_final_det      = gemini_det
                    st.session_state.rt_capture_pending = False
                    st.session_state.rt_capture_frame  = None
                    st.success(f"✅ Gemini: **{match}** → {lv_cfg['label']} ({lv_cfg['emoji']} {gula}g/100ml)")
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.rerun()
                else:
                    # Produk tidak ada di database, tawarkan pilih manual
                    conf_g = hasil_g.get("confidence", 0)
                    st.warning(
                        f"⚠️ Gemini mengenali **{g_name}** (conf {conf_g*100:.0f}%) "
                        "tapi tidak ada di database. Pilih manual:"
                    )
                    manual_pick = st.selectbox(
                        "Produk:", list(SUGAR_DB.keys()), key="sel_capture_manual"
                    )
                    if st.button("✅ Gunakan Pilihan Manual", key="btn_capture_manual", use_container_width=True):
                        manual_det = {
                            manual_pick: build_manual_product_result(manual_pick, source="Manual", confidence=0.0, csv_df=csv_df)
                        }
                        st.session_state.rt_frozen          = True
                        st.session_state.rt_frozen_frame    = st.session_state.rt_capture_frame.copy()
                        st.session_state.rt_final_det       = manual_det
                        st.session_state.rt_capture_pending = False
                        st.session_state.rt_capture_frame   = None
                        st.rerun()
                    st.markdown('</div>', unsafe_allow_html=True)
            else:
                # Gemini tidak dapat mengenali produk
                st.error(
                    f"❌ Gemini tidak dapat mengenali produk.\n"
                    f"**Detail:** {hasil_g.get('error','–')}"
                )
                st.markdown('</div>', unsafe_allow_html=True)
                if st.button("🔄 Coba Capture Lagi", key="btn_retry_capture", use_container_width=True):
                    st.session_state.rt_capture_pending = False
                    st.session_state.rt_capture_frame = None
                    st.rerun()

    # ── LOOP KAMERA ───────────────────────────────────────────────────────────
    if run and not st.session_state.rt_frozen and not st.session_state.rt_show_gemini_btn:
        cap      = cv2.VideoCapture(0)
        fps_buf  = []
        stab_buf = []

        while run:
            t0  = time.perf_counter()
            ret, frame = cap.read()
            if not ret:
                status_box.error("❌ Tidak dapat membaca kamera.")
                break

            results   = yolo_model(frame, verbose=False)
            high_det  = {}
            low_boxes = []

            for det in results[0].boxes:
                xyxy = det.xyxy.cpu().numpy().squeeze().astype(int)
                if not (xyxy.ndim == 1 and len(xyxy) == 4):
                    continue
                xmin, ymin, xmax, ymax = xyxy
                cls_id    = int(det.cls.item())
                classname = yolo_model.names[cls_id]
                conf      = det.conf.item()

                if conf >= yolo_thresh and classname in SUGAR_DB:
                    gula            = SUGAR_DB[classname]
                    lv_kode, lv_cfg = get_nutri_level(gula)
                    meta            = PRODUCT_META.get(classname, {"netto":250,"kalori":0})
                    total_gula      = round(gula * meta["netto"] / 100, 1)
                    akg             = round((total_gula / AKG_GULA) * 100, 1)
                    color           = lv_cfg["cv2_bgr"]

                    high_det[classname] = {
                        "gula": gula, "akg": akg,
                        "level_kode": lv_kode, "level_cfg": lv_cfg,
                        "kalori": meta["kalori"], "source": "YOLO",
                    }
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 3)
                    cv2.putText(frame, f"{classname} [{lv_kode}] {conf*100:.0f}%",
                                (xmin, max(ymin-10,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    cv2.putText(frame, f"{gula}g/100ml | AKG:{akg}%",
                                (xmin, ymax+22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif conf < yolo_thresh:
                    gray = (150, 150, 150)
                    cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), gray, 2)
                    cv2.putText(frame, f"? {conf*100:.0f}%",
                                (xmin, max(ymin-10,15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, gray, 2)
                    low_boxes.append((xmin, ymin, xmax, ymax, conf, classname))

            # ── Check jika tombol Capture ditekan ─────────────────────────────────
            if st.session_state.rt_capture_pending:
                st.session_state.rt_capture_frame = frame.copy()
                cap.release()
                st.rerun()
                break

            # ── Tidak ada bounding box → tunggu beberapa frame sebelum reset buffer ───
            if len(results[0].boxes) == 0:
                # Tambah counter untuk tunggu 15 frame sebelum reset stabilisasi
                if len(stab_buf) > 0:
                    st.session_state.rt_nodetect_counter += 1
                    # Jika sudah 15 frame tanpa deteksi, reset buffer
                    if st.session_state.rt_nodetect_counter >= 15:
                        stab_buf.clear()
                        st.session_state.rt_nodetect_counter = 0
                else:
                    st.session_state.rt_nodetect_counter = 0
                
                fps = 1 / max(time.perf_counter() - t0, 1e-6)
                fps_buf.append(fps)
                if len(fps_buf) > 30: fps_buf.pop(0)
                cv2.putText(frame, f"FPS: {np.mean(fps_buf):.1f}",
                            (10,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)
                cv2.putText(frame, "Arahkan ke produk minuman...",
                            (10, frame.shape[0]-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180,180,180), 1)
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                status_box.info("🔍 Mencari produk... arahkan kamera ke minuman kemasan")
                time.sleep(0.03)
                continue

            # ── Ada box confident → stabilisasi → freeze ──────────────────────
            if high_det:
                # Reset counter no-detection jika ada deteksi confident
                st.session_state.rt_nodetect_counter = 0
                
                detected_names = frozenset(high_det.keys())
                stab_buf.append(detected_names)
                if len(stab_buf) > stabilize_n:
                    stab_buf.pop(0)

                is_stable = (
                    len(stab_buf) >= stabilize_n and
                    all(s == stab_buf[0] for s in stab_buf)
                )

                stable_count = len([s for s in stab_buf if s == stab_buf[-1]])
                bar_w = int((stable_count / stabilize_n) * frame.shape[1])
                cv2.rectangle(frame, (0, frame.shape[0]-8), (bar_w, frame.shape[0]), (0,210,120), -1)
                cv2.putText(frame, f"Stabil: {stable_count}/{stabilize_n}",
                            (10, frame.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,220,130), 1)

                fps = 1 / max(time.perf_counter() - t0, 1e-6)
                fps_buf.append(fps)
                if len(fps_buf) > 30: fps_buf.pop(0)
                cv2.putText(frame, f"FPS: {np.mean(fps_buf):.1f}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                status_box.warning(f"⏳ Mendeteksi stabil... ({stable_count}/{stabilize_n} frame)")

                if is_stable:
                    if st.session_state.rt_freeze_timer is None:
                        st.session_state.rt_freeze_timer = time.time()

                    elapsed = time.time() - st.session_state.rt_freeze_timer
                    cv2.putText(frame, f"Freeze dalam: {max(freeze_delay - elapsed, 0):.1f}s",
                                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
                    frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    if elapsed >= freeze_delay:
                        st.session_state.rt_frozen         = True
                        st.session_state.rt_frozen_frame   = frame.copy()
                        st.session_state.rt_final_det      = high_det.copy()
                        st.session_state.rt_freeze_timer   = None
                        # aktifkan tombol koreksi Gemini
                        st.session_state.rt_show_gemini_corr = True

                        render_rt_hasil(high_det)
                        cap.release()
                        st.rerun()
                        break
                else:
                    st.session_state.rt_freeze_timer = None

            # ── Ada box tapi semua low-conf → terus scan, tidak tampilkan panel ──
            elif low_boxes and not high_det:
                # Jika ada low-confidence boxes, terus scan tanpa menampilkan panel Gemini
                fps = 1 / max(time.perf_counter() - t0, 1e-6)
                fps_buf.append(fps)
                if len(fps_buf) > 30: fps_buf.pop(0)
                cv2.putText(frame, f"FPS: {np.mean(fps_buf):.1f}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
                frame_window.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                status_box.info(
                    "⏳ YOLO ragu dengan deteksi ini. Coba perbaiki sudut/pencahayaan frame..."
                )
                time.sleep(0.03)
                continue

            time.sleep(0.03)

        try:
            cap.release()
        except Exception:
            pass

    # ── Manual override ────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📋 Tidak Terdeteksi? Pilih Produk Manual", expanded=False):
        st.caption("Jika YOLO dan Gemini gagal, pilih produk dari YOLO-DB secara manual:")
        manual_rt = st.selectbox("Produk:", list(SUGAR_DB.keys()), key="rt_manual")
        if st.button("✅ Analisis Produk Ini", key="rt_manual_btn", use_container_width=True):
            manual_det = {
                manual_rt: build_manual_product_result(manual_rt, source="Manual", confidence=0.0, csv_df=csv_df)
            }
            st.session_state.rt_frozen    = True
            st.session_state.rt_final_det = manual_det
            render_rt_hasil(manual_det)
            st.rerun()


# ════════════════════════════════════════════════════════════════════
# MODE 3 — MANUAL SELECTION (MENU)
# ════════════════════════════════════════════════════════════════════
elif mode == MANUAL_MODE:
    st.markdown('<div class="ns-section-title">📝 Pilih Produk Manual</div>', unsafe_allow_html=True)

    st.markdown(
        "Pilih produk secara manual dari database CSV (jika tersedia) atau dari YOLO-DB.")

    # Build choices: prefer CSV names (richer DB), fallback to SUGAR_DB
    csv_choices = csv_df["name"].tolist() if not csv_df.empty else []
    choices = csv_choices if csv_choices else list(SUGAR_DB.keys())

    search_q = st.text_input("Cari produk (nama):", value="", help="Ketik sebagian nama untuk memfilter")
    if search_q:
        filtered = [c for c in choices if search_q.lower() in c.lower()]
    else:
        filtered = choices

    if not filtered:
        st.warning("Tidak ada produk cocok untuk query ini.")
    else:
        manual_choice = st.selectbox("Pilih produk:", filtered, key="manual_menu_pick")
        if st.button("🔍 Analisis Manual", key="manual_menu_analyze", use_container_width=True):
            found_product = build_manual_product_result(manual_choice, source="Manual", confidence=0.0, csv_df=csv_df)
            if has_vector_db and embedder is not None and vectors is not None:
                found_product["all_similar"] = find_similar(manual_choice, embedder, vectors, csv_names, csv_df, top_k=5)
            # Render results
            render_summary_panel(found_product)
            render_upload_analysis(found_product, found_product.get("all_similar"))