# Camera Capture

Dieses Projekt stellt eine kleine Tk-Anwendung bereit, die zwei Kameras gleichzeitig
anzeigt, invertierte Graustufen berechnet, Canny-Kanten hervorhebt und – sofern eine
Stereo-Kalibrierung vorliegt – eine 3D-Tiefenansicht erzeugt.

## Stereo-Kalibrierung durchführen

1. **Kalibrierung starten**
   Öffne die Anwendung (`python -m dual_invert_app`) und stelle sicher, dass beide
   Kameras sichtbar sind. Drücke anschließend im linken Bedienfeld auf
   **„Stereo Kalibrierung“**. Die Anwendung startet die Kameras automatisch,
   falls sie zuvor gestoppt waren.

2. **Schachbrett bewegen**
   Halte das 6×6-Schachbrett nacheinander in verschiedene Positionen und
   Blickrichtungen. Sobald beide Kameras ein gültiges Muster erkennen, zeichnet
   die Anwendung ein Bildpaar auf und zeigt den Fortschritt im Statusfeld an.
   Ziel sind 18 gültige Paare (mindestens 12 sind erforderlich).

3. **Automatisches Speichern**
   Nach erfolgreicher Auswertung speichert die Anwendung
   `stereo_calibration.json` im Projektstammverzeichnis und legt sämtliche
   verwendeten Bilder im Ordner `calibration_captures/<timestamp>/` ab.
   Anschließend steht die Stereo-Ansicht sofort zur Verfügung.

### Alternative: Kalibrierung per Skript

Falls du lieber offline mit bereits vorhandenen Bildern arbeiten möchtest,
kannst du weiterhin `stereo_reconstruction.py` direkt verwenden. Sammle deine
Bildpaare in einem Ordner und führe beispielsweise folgendes Snippet aus:

```python
from pathlib import Path

import cv2
import stereo_reconstruction as sr

pairs = []
for i in range(1, 21):
    left = cv2.imread(f"calib/left_{i:02d}.png")
    right = cv2.imread(f"calib/right_{i:02d}.png")
    if left is None or right is None:
        continue
    pairs.append((left, right))

calibration = sr.compute_stereo_calibration(pairs, pattern_size=(6, 6), square_size=1.0)
sr.save_calibration(Path("stereo_calibration.json"), calibration)
```

Die erzeugte Datei wird von der Anwendung beim nächsten Start automatisch
geladen.

## Live-Bedienung

* **Start / Stop** – Öffnet bzw. schließt die konfigurierten Kameras.
* **Speichern** – Schreibt alle zuletzt angezeigten Ansichten in den Ordner
  `captures/`.
* **Refresh Auto Fokus** – Versucht, den Autofokus der Kameras neu zu triggern.
* **Beenden** – Schließt die Anwendung.

Das Originalbild wird mit ca. 30 FPS aktualisiert. Die Canny-Kanten sowie die
Stereo-Rekonstruktion laufen aus Performance-Gründen auf ~6–7 FPS.
