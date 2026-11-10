import streamlit as st
import pandas as pd
import hashlib
import datetime
import os

# ── DATABASE ──────────────────────────────────────────────────────────────────
from database import (
    init_db, get_all_produk, insert_produk,
    update_produk, delete_produk, sync_to_csv, import_from_csv
)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NutriScan — Admin Panel",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ══════════════════════════════════════════════════════════════════════════════
# NUTRI-LEVEL CONFIG (dibutuhkan untuk kalkulasi level)
# ══════════════════════════════════════════════════════════════════════════════
NUTRI_LEVEL_CONFIG = {
    "A": {"label": "A", "kategori": "Pilihan Lebih Sehat",  "hex": "#00A651", "emoji": "🟢", "range": "≤ 1 g/100ml"},
    "B": {"label": "B", "kategori": "Rendah Gula",          "hex": "#8BC53F", "emoji": "🟩", "range": "> 1 – 5 g/100ml"},
    "C": {"label": "C", "kategori": "Sedang",                "hex": "#F9A01B", "emoji": "🟠", "range": "> 5 – 10 g/100ml"},
    "D": {"label": "D", "kategori": "Tinggi Gula",           "hex": "#ED1C24", "emoji": "🔴", "range": "> 10 g/100ml"},
}

def get_nutri_level(gula: float) -> tuple:
    if gula <= 1:    return "A", NUTRI_LEVEL_CONFIG["A"]
    elif gula <= 5:  return "B", NUTRI_LEVEL_CONFIG["B"]
    elif gula <= 10: return "C", NUTRI_LEVEL_CONFIG["C"]
    else:            return "D", NUTRI_LEVEL_CONFIG["D"]

# ══════════════════════════════════════════════════════════════════════════════
# AUTENTIKASI
# ══════════════════════════════════════════════════════════════════════════════
ADMIN_CREDENTIALS = {
    "admin": hashlib.sha256("admin".encode()).hexdigest(),
}

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def check_login(username: str, password: str) -> bool:
    hashed = hash_password(password)
    return ADMIN_CREDENTIALS.get(username) == hashed

# ══════════════════════════════════════════════════════════════════════════════
# SESSION STATE INIT
# ══════════════════════════════════════════════════════════════════════════════
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "admin_user" not in st.session_state:
    st.session_state.admin_user = ""
if "login_time" not in st.session_state:
    st.session_state.login_time = None
if "activity_log" not in st.session_state:
    st.session_state.activity_log = []

def log_activity(action: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.activity_log.append({
        "waktu": ts,
        "user": st.session_state.admin_user,
        "aksi": action
    })

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* Topbar gradient */
.adm-topbar {
    height: 4px;
    background: linear-gradient(90deg, #185FA5 0%, #378ADD 50%, #00A651 100%);
    border-radius: 0;
    margin-bottom: 0;
}
/* Login card */
.login-wrapper {
    display: flex;
    justify-content: center;
    align-items: center;
    min-height: 60vh;
}
/* Badge status */
.adm-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 11px;
    font-weight: 600;
    padding: 4px 12px;
    border-radius: 99px;
    border: 1px solid;
}
.adm-badge-online {
    background: #E8F5E9;
    color: #1B5E20;
    border-color: #2E7D32;
}
/* Stat card */
.stat-card {
    background: rgba(128,128,128,0.05);
    border: 0.5px solid rgba(128,128,128,0.2);
    border-radius: 12px;
    padding: 16px;
    text-align: center;
}
.stat-num {
    font-size: 2em;
    font-weight: 700;
    line-height: 1;
}
.stat-label {
    font-size: 11px;
    opacity: 0.5;
    margin-top: 4px;
}
/* Level colors */
.lv-a { color: #00A651; }
.lv-b { color: #8BC53F; }
.lv-c { color: #F9A01B; }
.lv-d { color: #ED1C24; }
/* Table style */
[data-testid="stDataFrame"] { border-radius: 10px; overflow: hidden; }
/* Tombol style */
.stButton > button { border-radius: 8px !important; font-weight: 600 !important; }
/* Warning box */
.delete-warning {
    background: #FFF3E0;
    border: 1px solid #F57C00;
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 13px;
    color: #E65100;
}
/* Info box session */
.session-info {
    background: rgba(24, 95, 165, 0.05);
    border: 0.5px solid rgba(24, 95, 165, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 12px;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN LOGIN
# ══════════════════════════════════════════════════════════════════════════════
def render_login():
    st.markdown('<div class="adm-topbar"></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_l, col_mid, col_r = st.columns([1, 1.2, 1])
    with col_mid:
        # Header
        st.markdown("""
        <div style="text-align:center; margin-bottom: 28px;">
            <div style="font-size: 2.5em; margin-bottom: 8px;">🔐</div>
            <div style="font-size: 1.4em; font-weight: 700; letter-spacing: -0.3px;">NutriScan Admin</div>
            <div style="font-size: 0.85em; opacity: 0.5; margin-top: 4px;">Panel Manajemen Database Produk</div>
        </div>
        """, unsafe_allow_html=True)

        # Form login
        with st.container(border=True):
            st.markdown("#### 🔑 Login Administrator")
            st.markdown("<br>", unsafe_allow_html=True)

            username = st.text_input(
                "Username",
                placeholder="Masukkan username admin",
                key="login_username"
            )
            password = st.text_input(
                "Password",
                type="password",
                placeholder="Masukkan password",
                key="login_password"
            )

            st.markdown("<br>", unsafe_allow_html=True)
            login_btn = st.button("🚀 Masuk ke Panel Admin", type="primary", use_container_width=True)

            if login_btn:
                if not username or not password:
                    st.error("⚠️ Username dan password tidak boleh kosong.")
                elif check_login(username, password):
                    st.session_state.logged_in  = True
                    st.session_state.admin_user = username
                    st.session_state.login_time = datetime.datetime.now()
                    log_activity("Login berhasil")
                    st.success("✅ Login berhasil! Mengalihkan...")
                    st.rerun()
                else:
                    st.error("❌ Username atau password salah.")

        st.markdown("""
        <div style="text-align:center; font-size: 11px; opacity: 0.4; margin-top: 16px;">
            NutriScan Admin Panel · Hanya untuk Administrator
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HALAMAN UTAMA ADMIN (setelah login)
# ══════════════════════════════════════════════════════════════════════════════
def render_admin():
    init_db()

    # ── TOP BAR ──────────────────────────────────────────────────────────────
    st.markdown('<div class="adm-topbar"></div>', unsafe_allow_html=True)

    # ── HEADER ───────────────────────────────────────────────────────────────
    col_h1, col_h2, col_h3 = st.columns([3, 1.5, 1])

    with col_h1:
        st.markdown("""
        <div style="display:flex; align-items:center; gap:10px; padding: 8px 0;">
            <span style="font-size:1.8em">🗄️</span>
            <div>
                <div style="font-size:1.2em; font-weight:700; letter-spacing:-0.3px">NutriScan Admin Panel</div>
                <div style="font-size:0.78em; opacity:0.45">Manajemen Database Produk Minuman Kemasan</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_h2:
        login_time_str = st.session_state.login_time.strftime("%H:%M:%S") if st.session_state.login_time else "-"
        st.markdown(f"""
        <div class="session-info" style="margin-top:8px">
            <div>👤 <strong>{st.session_state.admin_user}</strong></div>
            <div style="opacity:0.6; font-size:11px">Login: {login_time_str}</div>
        </div>
        """, unsafe_allow_html=True)

    with col_h3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🚪 Logout", use_container_width=True):
            log_activity("Logout")
            st.session_state.logged_in  = False
            st.session_state.admin_user = ""
            st.session_state.login_time = None
            st.rerun()

    st.markdown("---")

    # ── STATISTIK RINGKAS ─────────────────────────────────────────────────────
    df_all = get_all_produk()

    total   = len(df_all)
    lv_counts = {"A": 0, "B": 0, "C": 0, "D": 0}
    if not df_all.empty:
        df_all["gula_per_100ml"] = df_all.apply(
            lambda r: round((float(r["total_gula"]) / float(r["netto"])) * 100, 2)
            if float(r["netto"]) > 0 else 0.0, axis=1
        )
        for g in df_all["gula_per_100ml"]:
            lv, _ = get_nutri_level(float(g))
            lv_counts[lv] += 1

    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    cards = [
        (sc1, "📦", str(total),        "Total Produk",     ""),
        (sc2, "🟢", str(lv_counts["A"]), "Level A — Sehat",  "#00A651"),
        (sc3, "🟩", str(lv_counts["B"]), "Level B — Rendah", "#8BC53F"),
        (sc4, "🟠", str(lv_counts["C"]), "Level C — Sedang", "#F9A01B"),
        (sc5, "🔴", str(lv_counts["D"]), "Level D — Tinggi", "#ED1C24"),
    ]
    for col, icon, num, label, color in cards:
        with col:
            clr = color if color else "var(--text-color)"
            st.markdown(f"""
            <div class="stat-card">
                <div style="font-size:1.4em">{icon}</div>
                <div class="stat-num" style="color:{clr}">{num}</div>
                <div class="stat-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS UTAMA ────────────────────────────────────────────────────────────
    tab_list, tab_add, tab_edit, tab_import, tab_log = st.tabs([
        "📋 Daftar Produk",
        "➕ Tambah Produk",
        "✏️ Edit / Hapus",
        "📥 Import & Export",
        "🕒 Log Aktivitas",
    ])

    # ══════════════════════════════════════════════════════════════════
    # TAB 1 — DAFTAR PRODUK
    # ══════════════════════════════════════════════════════════════════
    with tab_list:
        st.markdown("### 📋 Seluruh Produk dalam Database")

        if df_all.empty:
            st.info("💡 Belum ada produk dalam database. Tambah produk di tab **Tambah Produk**.")
        else:
            # Filter & Search
            col_s1, col_s2, col_s3 = st.columns([2, 1, 1])
            with col_s1:
                search_q = st.text_input("🔍 Cari produk", placeholder="Ketik nama produk...")
            with col_s2:
                filter_lv = st.selectbox("Filter Level", ["Semua", "A", "B", "C", "D"])
            with col_s3:
                filter_pkg = st.selectbox("Filter Kemasan", ["Semua"] + sorted(df_all["packaging"].unique().tolist()))

            # Terapkan filter
            df_view = df_all.copy()
            if search_q:
                df_view = df_view[df_view["name"].str.contains(search_q, case=False, na=False)]
            if filter_lv != "Semua":
                df_view = df_view[df_view["gula_per_100ml"].apply(lambda g: get_nutri_level(g)[0]) == filter_lv]
            if filter_pkg != "Semua":
                df_view = df_view[df_view["packaging"] == filter_pkg]

            # Tambah kolom level
            df_view = df_view.copy()
            df_view["Level"] = df_view["gula_per_100ml"].apply(
                lambda g: f"{get_nutri_level(g)[1]['emoji']} {get_nutri_level(g)[0]}"
            )

            tampil_cols = ["id", "name", "packaging", "netto", "total_gula",
                           "gula_per_100ml", "kalori", "jenis", "Level"]
            st.dataframe(
                df_view[tampil_cols].rename(columns={
                    "id": "ID", "name": "Nama Produk", "packaging": "Kemasan",
                    "netto": "Netto (ml)", "total_gula": "Total Gula (g)",
                    "gula_per_100ml": "Gula/100ml (g)", "kalori": "Kalori (kkal)",
                    "jenis": "Jenis"
                }),
                use_container_width=True,
                hide_index=True,
                height=420
            )
            st.caption(f"Menampilkan **{len(df_view)}** dari **{total}** produk")

        # Sinkron ke CSV
        st.markdown("---")
        col_sync1, col_sync2 = st.columns([2, 1])
        with col_sync1:
            st.markdown("**🔄 Sinkronisasi Database → CSV**")
            st.caption("Export seluruh data SQLite ke file CSV (carimanis_botol.csv & carimanis_kaleng.csv)")
        with col_sync2:
            if st.button("🔄 Sync ke CSV", use_container_width=True, type="primary"):
                n = sync_to_csv()
                st.cache_data.clear()
                log_activity(f"Sync DB → CSV ({n} produk)")
                st.success(f"✅ {n} produk berhasil di-export ke CSV.")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════
    # TAB 2 — TAMBAH PRODUK
    # ══════════════════════════════════════════════════════════════════
    with tab_add:
        st.markdown("### ➕ Tambah Produk Baru")

        col_form, col_preview = st.columns([1.2, 1], gap="large")

        with col_form:
            with st.form("form_tambah_admin", clear_on_submit=True):
                st.markdown("#### 📝 Isi Data Produk")

                inp_name = st.text_input(
                    "Nama Produk *",
                    placeholder="cth: Pocari Sweat Botol 500ml"
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    inp_netto     = st.number_input("Netto (ml) *",    min_value=1.0,  value=250.0, step=5.0)
                    inp_kalori    = st.number_input("Kalori (kkal)",    min_value=0.0,  value=0.0,   step=1.0)
                with col_b:
                    inp_total_gula = st.number_input("Total Gula (g) *", min_value=0.0, value=0.0,   step=0.5)
                    inp_packaging  = st.selectbox(
                        "Jenis Kemasan",
                        ["botol", "kaleng", "kotak", "sachet", "lainnya"]
                    )

                inp_jenis = st.text_input("Jenis Minuman", value="Minuman Kemasan")
                inp_merek = st.text_input("Merek (opsional)", placeholder="cth: Pocari Sweat")

                st.markdown("<br>", unsafe_allow_html=True)
                submitted = st.form_submit_button("💾 Simpan Produk", type="primary", use_container_width=True)

        with col_preview:
            st.markdown("#### 👁️ Preview Nutri-Level")
            if "inp_netto" in dir():
                pass

            # Preview real-time (di luar form)
            prev_netto = 250.0
            prev_total_gula = 0.0
            try:
                prev_netto      = st.session_state.get("_prev_netto",      250.0)
                prev_total_gula = st.session_state.get("_prev_total_gula", 0.0)
            except Exception:
                pass

            st.markdown("""
            <div style="background:rgba(128,128,128,0.05);border:0.5px solid rgba(128,128,128,0.2);
                        border-radius:12px;padding:20px;text-align:center;margin-top:8px">
                <div style="font-size:0.85em;opacity:0.5;margin-bottom:12px">
                    Isi form di sebelah kiri untuk melihat preview level nutrisi
                </div>
                <div style="font-size:3em">🧪</div>
                <div style="font-size:0.85em;margin-top:8px;opacity:0.6">
                    Level akan dihitung otomatis berdasarkan<br>
                    <strong>Total Gula ÷ Netto × 100</strong>
                </div>
                <div style="margin-top:16px;font-size:12px;opacity:0.45">
                    A: ≤1g · B: 1–5g · C: 5–10g · D: >10g
                    <br><span style="font-size:10px">(per 100ml)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

        if submitted:
            if not inp_name.strip():
                st.error("❌ Nama produk tidak boleh kosong.")
            elif inp_netto <= 0:
                st.error("❌ Netto harus lebih dari 0.")
            else:
                nama_final = inp_name.strip()
                ok = insert_produk(
                    nama_final, inp_netto, inp_total_gula,
                    inp_kalori, inp_packaging, inp_jenis
                )
                if ok:
                    sync_to_csv()
                    st.cache_data.clear()
                    g_per_100 = round((inp_total_gula / inp_netto) * 100, 2)
                    lv, cfg   = get_nutri_level(g_per_100)
                    log_activity(f"Tambah produk: {nama_final} (Level {lv})")
                    st.success(
                        f"✅ **'{nama_final}'** berhasil ditambahkan! "
                        f"{cfg['emoji']} Level **{lv}** — {g_per_100}g/100ml"
                    )
                    st.rerun()
                else:
                    st.error(f"❌ Produk **'{inp_name}'** sudah ada di database.")

    # ══════════════════════════════════════════════════════════════════
    # TAB 3 — EDIT / HAPUS
    # ══════════════════════════════════════════════════════════════════
    with tab_edit:
        st.markdown("### ✏️ Edit atau Hapus Produk")

        if df_all.empty:
            st.info("💡 Database masih kosong.")
        else:
            # Pilih produk
            prod_options = df_all["name"].tolist()
            pilihan = st.selectbox(
                "🔎 Pilih produk yang ingin diedit/dihapus:",
                prod_options,
                key="edit_admin_select"
            )
            row = df_all[df_all["name"] == pilihan].iloc[0]

            # Info produk saat ini
            g_cur = float(row["gula_per_100ml"]) if "gula_per_100ml" in row else 0.0
            lv_cur, cfg_cur = get_nutri_level(g_cur)

            col_info1, col_info2, col_info3, col_info4 = st.columns(4)
            with col_info1:
                st.metric("Netto", f"{row['netto']} ml")
            with col_info2:
                st.metric("Total Gula", f"{row['total_gula']} g")
            with col_info3:
                st.metric("Gula/100ml", f"{g_cur:.2f} g")
            with col_info4:
                st.metric("Level Saat Ini", f"{cfg_cur['emoji']} {lv_cur}")

            st.markdown("---")

            # Form edit
            with st.form("form_edit_admin"):
                st.markdown("#### 📝 Edit Data Produk")
                col_e1, col_e2 = st.columns(2)
                pkg_opts = ["botol", "kaleng", "kotak", "sachet", "lainnya"]
                with col_e1:
                    e_name      = st.text_input("Nama",           value=str(row["name"]))
                    e_netto     = st.number_input("Netto (ml)",   value=float(row["netto"]),      min_value=1.0, step=5.0)
                    e_packaging = st.selectbox(
                        "Kemasan", pkg_opts,
                        index=pkg_opts.index(row["packaging"]) if row["packaging"] in pkg_opts else 0
                    )
                with col_e2:
                    e_total_gula = st.number_input("Total Gula (g)",  value=float(row["total_gula"]), step=0.5)
                    e_kalori     = st.number_input("Kalori (kkal)",   value=float(row["kalori"]),     step=1.0)
                    e_jenis      = st.text_input("Jenis",             value=str(row["jenis"]))

                # Preview level baru
                if e_netto > 0:
                    new_g = round((e_total_gula / e_netto) * 100, 2)
                    lv_new, cfg_new = get_nutri_level(new_g)
                    st.info(
                        f"Preview setelah edit: **{new_g} g/100ml** → "
                        f"{cfg_new['emoji']} Level **{lv_new}** — {cfg_new['kategori']}"
                    )

                st.markdown("<br>", unsafe_allow_html=True)
                btn_update = st.form_submit_button("💾 Update Produk", type="primary", use_container_width=True)

            # Tombol hapus (di luar form agar tidak conflict)
            st.markdown("---")
            with st.container(border=True):
                st.markdown(f"""
                <div class="delete-warning">
                    ⚠️ <strong>Zona Berbahaya</strong> — Hapus produk <em>"{pilihan}"</em> secara permanen dari database.
                    Tindakan ini tidak dapat dibatalkan.
                </div>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                confirm_del = st.checkbox(
                    f'Saya yakin ingin menghapus **"{pilihan}"**',
                    key="confirm_delete_chk"
                )
                btn_del = st.button(
                    "🗑️ Hapus Produk Ini",
                    disabled=not confirm_del,
                    use_container_width=True,
                    key="btn_hapus_produk"
                )

            if btn_update:
                update_produk(
                    int(row["id"]), e_name, e_netto,
                    e_total_gula, e_kalori, e_packaging, e_jenis
                )
                sync_to_csv()
                st.cache_data.clear()
                log_activity(f"Update produk: {pilihan} → {e_name}")
                st.success(f"✅ Produk **'{e_name}'** berhasil diperbarui.")
                st.rerun()

            if btn_del and confirm_del:
                delete_produk(int(row["id"]))
                sync_to_csv()
                st.cache_data.clear()
                log_activity(f"Hapus produk: {pilihan}")
                st.success(f"🗑️ Produk **'{pilihan}'** berhasil dihapus dari database.")
                st.rerun()

    # ══════════════════════════════════════════════════════════════════
    # TAB 4 — IMPORT & EXPORT
    # ══════════════════════════════════════════════════════════════════
    with tab_import:
        st.markdown("### 📥 Import & Export Data")

        col_imp, col_exp = st.columns(2, gap="large")

        # ── IMPORT ──────────────────────────────────────────────────
        with col_imp:
            with st.container(border=True):
                st.markdown("#### 📤 Import dari CSV")
                st.caption(
                    "Upload file CSV dengan kolom: `name`, `netto`, `total_gula`, "
                    "`kalori`, `packaging`, `jenis`"
                )

                uploaded_csv = st.file_uploader(
                    "Pilih file CSV", type=["csv"], key="import_csv_upload"
                )

                if uploaded_csv:
                    try:
                        df_imp = pd.read_csv(uploaded_csv)
                        st.dataframe(df_imp.head(10), use_container_width=True, hide_index=True)
                        st.caption(f"Preview: {len(df_imp)} baris")

                        required_cols = {"name", "netto", "total_gula"}
                        missing = required_cols - set(df_imp.columns)
                        if missing:
                            st.error(f"❌ Kolom wajib tidak ditemukan: {missing}")
                        else:
                            if st.button("⬇️ Import ke Database", type="primary", use_container_width=True):
                                # Simpan sementara dan import
                                tmp_path = "/tmp/nutriscan_import_tmp.csv"
                                df_imp.to_csv(tmp_path, index=False)
                                n = import_from_csv(tmp_path)
                                sync_to_csv()
                                st.cache_data.clear()
                                log_activity(f"Import CSV: {n} produk dari {uploaded_csv.name}")
                                st.success(f"✅ **{n}** produk berhasil diimport dari {uploaded_csv.name}!")
                                st.rerun()
                    except Exception as e:
                        st.error(f"❌ Gagal membaca CSV: {e}")
                else:
                    st.info("📁 Upload file CSV di atas untuk memulai import.")

                st.markdown("---")
                st.markdown("**Import dari file CSV bawaan (carimanis):**")
                csv_files = ["carimanis_botol.csv", "carimanis_kaleng.csv"]
                for csv_f in csv_files:
                    if os.path.exists(csv_f):
                        if st.button(f"📂 Import {csv_f}", use_container_width=True, key=f"imp_{csv_f}"):
                            n = import_from_csv(csv_f)
                            sync_to_csv()
                            st.cache_data.clear()
                            log_activity(f"Import file bawaan: {csv_f} ({n} produk)")
                            st.success(f"✅ {n} produk dari **{csv_f}** berhasil diimport.")
                            st.rerun()
                    else:
                        st.caption(f"⚠️ `{csv_f}` tidak ditemukan di direktori.")

        # ── EXPORT ──────────────────────────────────────────────────
        with col_exp:
            with st.container(border=True):
                st.markdown("#### 📥 Export ke CSV")
                st.caption("Download seluruh data database dalam format CSV.")

                if not df_all.empty:
                    # Export semua
                    csv_all = df_all.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="⬇️ Download Semua Data (CSV)",
                        data=csv_all,
                        file_name=f"nutriscan_export_{datetime.date.today()}.csv",
                        mime="text/csv",
                        use_container_width=True,
                        type="primary"
                    )

                    st.markdown("---")
                    st.markdown("**Export per level nutrisi:**")
                    for lv_code in ["A", "B", "C", "D"]:
                        cfg_x = NUTRI_LEVEL_CONFIG[lv_code]
                        df_lv = df_all[
                            df_all["gula_per_100ml"].apply(lambda g: get_nutri_level(g)[0]) == lv_code
                        ]
                        count_lv = len(df_lv)
                        if count_lv > 0:
                            csv_lv = df_lv.to_csv(index=False).encode("utf-8")
                            st.download_button(
                                label=f"{cfg_x['emoji']} Level {lv_code} — {cfg_x['kategori']} ({count_lv} produk)",
                                data=csv_lv,
                                file_name=f"nutriscan_level_{lv_code}_{datetime.date.today()}.csv",
                                mime="text/csv",
                                use_container_width=True,
                                key=f"dl_lv_{lv_code}"
                            )
                        else:
                            st.caption(f"{cfg_x['emoji']} Level {lv_code}: tidak ada produk")

                    st.markdown("---")
                    # Sync ke CSV lokal
                    if st.button("🔄 Sync ke CSV Lokal", use_container_width=True):
                        n = sync_to_csv()
                        log_activity(f"Sync ke CSV lokal ({n} produk)")
                        st.success(f"✅ {n} produk berhasil di-sync ke CSV lokal.")
                else:
                    st.info("💡 Database masih kosong, tidak ada yang bisa di-export.")

    # ══════════════════════════════════════════════════════════════════
    # TAB 5 — LOG AKTIVITAS
    # ══════════════════════════════════════════════════════════════════
    with tab_log:
        st.markdown("### 🕒 Log Aktivitas Admin")
        st.caption("Riwayat aksi yang dilakukan dalam sesi ini.")

        if st.session_state.activity_log:
            df_log = pd.DataFrame(st.session_state.activity_log[::-1])  # terbaru di atas
            df_log.columns = ["Waktu", "User", "Aksi"]
            st.dataframe(df_log, use_container_width=True, hide_index=True, height=400)
            st.caption(f"Total {len(st.session_state.activity_log)} aktivitas dalam sesi ini.")

            # Download log
            log_csv = df_log.to_csv(index=False).encode("utf-8")
            st.download_button(
                "⬇️ Download Log Aktivitas",
                data=log_csv,
                file_name=f"nutriscan_log_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        else:
            st.info("📭 Belum ada aktivitas yang tercatat dalam sesi ini.")

    # ── FOOTER ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center; font-size:11px; opacity:0.35; padding:8px 0">
        NutriScan Admin Panel · Akses terbatas untuk Administrator
        · Data disimpan di SQLite lokal
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRYPOINT
# ══════════════════════════════════════════════════════════════════════════════
def main():
    if not st.session_state.logged_in:
        render_login()
    else:
        render_admin()

if __name__ == "__main__":
    main()