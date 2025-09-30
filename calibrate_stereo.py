from pathlib import Path
import cv2
import stereo_reconstruction as sr  # dein Modul

pairs_dir = Path("data/stereo_pairs")
left_prefix = "kamera0_"
right_prefix = "kamera1_"
extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

lefts = sorted([p for p in pairs_dir.iterdir() if p.suffix.lower() in extensions and p.name.startswith(left_prefix)])
rights = sorted([p for p in pairs_dir.iterdir() if p.suffix.lower() in extensions and p.name.startswith(right_prefix)])

if len(lefts) != len(rights) or len(lefts) == 0:
    raise RuntimeError(f"Anzahl linke ({len(lefts)}) vs. rechte ({len(rights)}) Bilder passt nicht oder ist 0.")

def index_of(name: str, prefix: str) -> str:
    core = name[len(prefix):]
    return core.split(".")[0]

left_map = {index_of(p.name, left_prefix): p for p in lefts}
right_map = {index_of(p.name, right_prefix): p for p in rights}
common_keys = sorted(set(left_map.keys()) & set(right_map.keys()))
if not common_keys:
    raise RuntimeError("Keine passenden Index-Paare gefunden (Namensschema prüfen).")

pairs_imgs = []
for k in common_keys:
    Lp = left_map[k]
    Rp = right_map[k]

    # 1) Farbig einlesen (empfohlen, weil _find_corners BGR->Gray macht)
    imgL = cv2.imread(str(Lp), cv2.IMREAD_COLOR)
    imgR = cv2.imread(str(Rp), cv2.IMREAD_COLOR)

    # Falls aus irgendeinem Grund doch 1-Kanal reinkommt: robust nach BGR konvertieren
    if imgL is None or imgR is None:
        raise RuntimeError(f"Konnte Bilder nicht lesen: {Lp} / {Rp}")
    if len(imgL.shape) == 2:
        imgL = cv2.cvtColor(imgL, cv2.COLOR_GRAY2BGR)
    if len(imgR.shape) == 2:
        imgR = cv2.cvtColor(imgR, cv2.COLOR_GRAY2BGR)

    if imgL.shape[:2] != imgR.shape[:2]:
        raise RuntimeError(f"Auflösungen ungleich bei Paar {k}: {imgL.shape} vs {imgR.shape}")

    pairs_imgs.append((imgL, imgR))

print(f"Gefundene, geprüfte Paare: {len(pairs_imgs)}")

# Optional: Falls dein sr.compute_stereo_calibration ein pattern_size braucht:
# calibration = sr.compute_stereo_calibration(pairs_imgs, pattern_size=(9,6))
calibration = sr.compute_stereo_calibration(pairs_imgs)

out_file = Path("stereo_calibration.json")
sr.save_calibration(out_file, calibration)
print(f"Kalibrierungsdatei geschrieben: {out_file.resolve()}")
