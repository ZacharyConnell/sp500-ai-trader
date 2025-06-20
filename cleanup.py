import os
import glob
import zipfile
import datetime
from pathlib import Path

# --- Configuration ---
DAYS_TO_KEEP = 30
MAX_BACKUPS = 5
VERBOSE = True
ZIP_OLD_CSVS = True

# --- Directory Paths ---
HISTORY_DIR = Path("data/history")
BACKUP_DIR = Path("models/backups")
EQUITY_DIR = Path("data/equity_curves")
LOG_DIR = Path("logs")
ARCHIVE_DIR = Path("data/archives")

def delete_old_files(path, pattern, days, zip_csvs=False):
    now = datetime.datetime.now(datetime.timezone.utc)
    cutoff = now - datetime.timedelta(days=days)
    removed = 0

    for file in glob.glob(str(path / pattern)):
        ftime = datetime.datetime.fromtimestamp(os.path.getmtime(file), tz=datetime.timezone.utc)
        if ftime < cutoff:
            if zip_csvs and file.endswith(".csv"):
                base = os.path.basename(file)
                # Include hour and minute to reduce collision chance
                date_str = ftime.strftime("%Y-%m-%d_%H-%M")
                zip_path = ARCHIVE_DIR / f"predictions_{date_str}.zip"
                os.makedirs(ARCHIVE_DIR, exist_ok=True)
                with zipfile.ZipFile(zip_path, 'a', zipfile.ZIP_DEFLATED) as zf:
                    zf.write(file, arcname=base)
                if VERBOSE:
                    print(f"🗜️ Zipped {file} → {zip_path}")
            os.remove(file)
            if VERBOSE:
                print(f"🧹 Removed: {file}")
            removed += 1
    return removed

def retain_recent_backups(directory, max_files):
    all_models = sorted(
        glob.glob(str(directory / "lstm_model_*.h5")),
        key=os.path.getmtime,
        reverse=True
    )
    to_delete = all_models[max_files:]
    for file in to_delete:
        os.remove(file)
        if VERBOSE:
            print(f"🗑️ Removed model backup: {file}")
    return len(to_delete)

if __name__ == "__main__":
    print("🧽 Starting cleanup...")

    # Ensure required directories exist
    for p in [HISTORY_DIR, BACKUP_DIR, EQUITY_DIR, LOG_DIR, ARCHIVE_DIR]:
        os.makedirs(p, exist_ok=True)

    n1 = delete_old_files(HISTORY_DIR, "*.csv", DAYS_TO_KEEP, zip_csvs=ZIP_OLD_CSVS)
    n2 = delete_old_files(EQUITY_DIR, "*.png", DAYS_TO_KEEP)
    n3 = delete_old_files(LOG_DIR, "*.log", DAYS_TO_KEEP) + delete_old_files(LOG_DIR, "*.txt", DAYS_TO_KEEP)
    n4 = retain_recent_backups(BACKUP_DIR, MAX_BACKUPS)

    print("✅ Cleanup complete:")
    print(f"   - {n1} history CSVs removed (zipped if enabled)")
    print(f"   - {n2} equity plots removed")
    print(f"   - {n3} old logs removed")
    print(f"   - {n4} model backups removed")