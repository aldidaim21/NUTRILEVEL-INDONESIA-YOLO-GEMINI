import sqlite3
import pandas as pd
import os

DB_PATH     = "nutriscan.db"
CSV_TARGETS = ["carimanis_botol.csv", "carimanis_kaleng.csv"]  # CSV yang akan di-overwrite saat sync

def _get_db_connection():
    """Helper untuk membuka koneksi dengan timeout dan WAL mode."""
    con = sqlite3.connect(DB_PATH, timeout=10.0)
    con.execute("PRAGMA journal_mode=WAL")
    return con


# ── 1. INISIALISASI ────────────────────────────────────────────────
def init_db():
    """Buat tabel jika belum ada. Dipanggil sekali saat app start."""
    con = _get_db_connection()
    try:
        con.execute("""
            CREATE TABLE IF NOT EXISTS produk (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                name        TEXT    NOT NULL UNIQUE,
                netto       REAL    NOT NULL DEFAULT 250,
                total_gula  REAL    NOT NULL DEFAULT 0,
                kalori      REAL             DEFAULT 0,
                packaging   TEXT             DEFAULT 'botol',
                jenis       TEXT             DEFAULT 'Minuman Kemasan',
                created_at  TEXT             DEFAULT (datetime('now','localtime'))
            )
        """)
        con.commit()
    finally:
        con.close()


# ── 2. READ ────────────────────────────────────────────────────────
def get_all_produk() -> pd.DataFrame:
    con = _get_db_connection()
    try:
        df  = pd.read_sql("SELECT * FROM produk ORDER BY name", con)
        if not df.empty:
            df["gula_per_100ml"] = df.apply(
                lambda r: round((r["total_gula"] / r["netto"]) * 100, 2)
                if r["netto"] > 0 else 0, axis=1
            )
        return df
    finally:
        con.close()


# ── 3. CREATE ──────────────────────────────────────────────────────
def insert_produk(name, netto, total_gula, kalori, packaging, jenis) -> bool:
    """Return True jika berhasil, False jika nama duplikat."""
    try:
        con = _get_db_connection()
        try:
            con.execute(
                "INSERT INTO produk (name,netto,total_gula,kalori,packaging,jenis) VALUES (?,?,?,?,?,?)",
                (name, netto, total_gula, kalori, packaging, jenis)
            )
            con.commit()
            return True
        finally:
            con.close()
    except sqlite3.IntegrityError:
        return False


# ── 4. UPDATE ──────────────────────────────────────────────────────
def update_produk(produk_id, name, netto, total_gula, kalori, packaging, jenis):
    con = _get_db_connection()
    try:
        con.execute(
            """UPDATE produk
               SET name=?, netto=?, total_gula=?, kalori=?, packaging=?, jenis=?
               WHERE id=?""",
            (name, netto, total_gula, kalori, packaging, jenis, produk_id)
        )
        con.commit()
    finally:
        con.close()


# ── 5. DELETE ──────────────────────────────────────────────────────
def delete_produk(produk_id):
    con = _get_db_connection()
    try:
        con.execute("DELETE FROM produk WHERE id=?", (produk_id,))
        con.commit()
    finally:
        con.close()


# ── 6. SYNC KE CSV ─────────────────────────────────────────────────
def sync_to_csv() -> int:
    """Ekspor semua data DB → CSV_TARGETS (botol & kaleng). Dipanggil setelah setiap perubahan."""
    df   = get_all_produk()
    cols = ["name", "netto", "total_gula", "kalori", "packaging", "jenis"]
    for csv_file in CSV_TARGETS:
        df[cols].to_csv(csv_file, index=False)
    return len(df)


# ── 7. SEED AWAL DARI CSV LAMA ─────────────────────────────────────
def import_from_csv(csv_path: str) -> int:
    """Baca CSV lama → insert ke DB. Dipanggil sekali saat DB masih kosong."""
    if not os.path.exists(csv_path):
        return 0
    df = pd.read_csv(csv_path)
    con = _get_db_connection()
    count = 0
    try:
        for _, row in df.iterrows():
            try:
                con.execute(
                    "INSERT INTO produk (name,netto,total_gula,kalori,packaging,jenis) VALUES (?,?,?,?,?,?)",
                    (
                        str(row.get("name", "")).strip(),
                        float(row.get("netto", 250)),
                        float(row.get("total_gula", 0)),
                        float(row.get("kalori", 0)),
                        str(row.get("packaging", "botol")),
                        str(row.get("jenis", "Minuman Kemasan")),
                    )
                )
                count += 1
            except sqlite3.IntegrityError:
                pass
        con.commit()
    finally:
        con.close()
    return count