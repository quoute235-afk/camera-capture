# Camera Capture

Dieses Projekt stellt eine kleine Tk-Anwendung bereit, die zwei Kameras gleichzeitig
anzeigt, invertierte Graustufen berechnet, Canny-Kanten hervorhebt und – sofern eine
Stereo-Kalibrierung vorliegt – eine 3D-Tiefenansicht erzeugt.

## Stereo-Kalibrierung durchführen

1. **Schachbrett aufnehmen**  
   Nimm pro Kamera mehrere synchronisierte Bilder des 6×6-Kalibrierfeldes auf.
   Die Bilder beider Kameras müssen jeweils dieselbe Auflösung besitzen.

2. **Kalibrierung berechnen**  
   Verwende das Modul `stereo_reconstruction.py`, um aus den Bildpaaren die
   Intrinsik und Extrinsik der Kameras zu bestimmen:

   ```python
   from pathlib import Path

   import cv2
   import stereo_reconstruction as sr

   # Beispiel: Bilder aus einem Ordner laden
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

   Das Script erkennt automatisch gültige Schachbrettpaare, optimiert die Ecke
   subpixelgenau und speichert alle Parameter samt Reprojektionsfehler.

3. **Datei ablegen**  
   Lege die erzeugte `stereo_calibration.json` im Projektstammverzeichnis ab.
   Die GUI lädt die Datei beim Start und zeigt in der ersten Kamerazeile ein
   zusätzliches Panel **„3D Ansicht“** an. Fehlt die Kalibrierung, bleibt das
   Panel mit einem Hinweis deaktiviert.

## Live-Bedienung

* **Start / Stop** – Öffnet bzw. schließt die konfigurierten Kameras.
* **Speichern** – Schreibt alle zuletzt angezeigten Ansichten in den Ordner
  `captures/`.
* **Refresh Auto Fokus** – Versucht, den Autofokus der Kameras neu zu triggern.
* **Beenden** – Schließt die Anwendung.

Das Originalbild wird mit ca. 30 FPS aktualisiert. Die Canny-Kanten sowie die
Stereo-Rekonstruktion laufen aus Performance-Gründen auf ~6–7 FPS.
