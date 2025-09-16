import cv2
import numpy as np
import threading
import time
from datetime import datetime
import os
import sys

def test_single_camera(index):
    """Testet eine einzelne Kamera ausführlich"""
    print(f"\n=== Test Kamera Index {index} ===")
    
    # Verschiedene Backends testen
    backends = [
        (cv2.CAP_DSHOW, "DirectShow"),
        (cv2.CAP_MSMF, "Media Foundation"),
        (cv2.CAP_ANY, "Auto")
    ]
    
    for backend, name in backends:
        print(f"Teste {name} Backend...")
        cap = cv2.VideoCapture(index, backend)
        
        if cap.isOpened():
            print(f"  ✓ {name} Backend funktioniert")
            
            # Frame testen
            ret, frame = cap.read()
            if ret and frame is not None:
                h, w = frame.shape[:2]
                print(f"  ✓ Frame erfolgreich: {w}x{h}")
                
                # 720p testen
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                
                actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                print(f"  720p Test: {actual_w}x{actual_h}")
                
                cap.release()
                return True, backend
            else:
                print(f"  ✗ Kein Frame von {name}")
        else:
            print(f"  ✗ {name} Backend fehlgeschlagen")
        
        cap.release()
    
    return False, None

def comprehensive_camera_scan():
    """Umfassender Kamera-Scan"""
    print("=== Umfassender Kamera-Scan ===")
    working_cameras = []
    
    for i in range(10):  # Teste mehr Indizes
        print(f"\nScanne Index {i}...")
        success, backend = test_single_camera(i)
        
        if success:
            working_cameras.append((i, backend))
            print(f"✓ Kamera {i} funktioniert mit Backend {backend}")
        else:
            print(f"✗ Kamera {i} nicht verfügbar")
    
    print(f"\n=== Scan Ergebnis ===")
    print(f"Gefundene Kameras: {len(working_cameras)}")
    for idx, backend in working_cameras:
        print(f"  Index {idx}: Backend {backend}")
    
    return working_cameras

class DebugDualCapture:
    def __init__(self, cam1_config, cam2_config):
        """
        Debug-Version mit ausführlicher Fehlerbehandlung
        cam1_config, cam2_config: Tuple (index, backend)
        """
        self.cam1_index, self.cam1_backend = cam1_config
        self.cam2_index, self.cam2_backend = cam2_config
        self.cap1 = None
        self.cap2 = None
        self.frame1 = None
        self.frame2 = None
        self.running = False
        
        print(f"Initialisiere mit:")
        print(f"  Kamera 1: Index {self.cam1_index}, Backend {self.cam1_backend}")
        print(f"  Kamera 2: Index {self.cam2_index}, Backend {self.cam2_backend}")
    
    def safe_camera_init(self, index, backend, name):
        """Sichere Kamera-Initialisierung mit Fehlerbehandlung"""
        print(f"\nInitialisiere {name}...")
        
        try:
            cap = cv2.VideoCapture(index, backend)
            
            if not cap.isOpened():
                print(f"  ✗ {name}: Kann nicht geöffnet werden")
                return None
            
            print(f"  ✓ {name}: Erfolgreich geöffnet")
            
            # Test Frame
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"  ✗ {name}: Kein Frame verfügbar")
                cap.release()
                return None
            
            print(f"  ✓ {name}: Frame OK ({frame.shape[1]}x{frame.shape[0]})")
            
            # 720p einstellen
            print(f"  Stelle 720p für {name} ein...")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Verifizieren
            time.sleep(0.5)  # Kurz warten
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  ✓ {name}: Eingestellt auf {actual_w}x{actual_h} @ {actual_fps}fps")
            
            # Finaler Frame-Test
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"  ✓ {name}: Finaler Test erfolgreich")
                return cap
            else:
                print(f"  ✗ {name}: Finaler Test fehlgeschlagen")
                cap.release()
                return None
                
        except Exception as e:
            print(f"  ✗ {name}: Exception - {str(e)}")
            return None
    
    def initialize_cameras(self):
        """Initialisiert beide Kameras mit Debug-Output"""
        print("\n=== Kamera Initialisierung ===")
        
        self.cap1 = self.safe_camera_init(self.cam1_index, self.cam1_backend, "Kamera 1")
        if not self.cap1:
            print("FEHLER: Kamera 1 Initialisierung fehlgeschlagen!")
            return False
        
        self.cap2 = self.safe_camera_init(self.cam2_index, self.cam2_backend, "Kamera 2")
        if not self.cap2:
            print("FEHLER: Kamera 2 Initialisierung fehlgeschlagen!")
            return False
        
        print("\n✓ Beide Kameras erfolgreich initialisiert!")
        
        # Auto-Fokus mit Debug
        print("\nAuto-Fokus Phase...")
        for i in range(10):
            ret1, _ = self.cap1.read()
            ret2, _ = self.cap2.read()
            if not ret1 or not ret2:
                print(f"  Frame-Verlust während Auto-Fokus: Cam1={ret1}, Cam2={ret2}")
            time.sleep(0.2)
        
        print("✓ Auto-Fokus abgeschlossen")
        return True
    
    def capture_thread_cam1(self):
        """Thread für Kamera 1 mit Fehlerbehandlung"""
        error_count = 0
        while self.running:
            if self.cap1 and self.cap1.isOpened():
                ret, frame = self.cap1.read()
                if ret and frame is not None:
                    self.frame1 = frame.copy()
                    error_count = 0  # Reset error counter
                else:
                    error_count += 1
                    if error_count % 30 == 0:  # Alle 30 Fehler melden
                        print(f"Kamera 1: {error_count} Frames verloren")
            time.sleep(0.033)  # ~30fps
    
    def capture_thread_cam2(self):
        """Thread für Kamera 2 mit Fehlerbehandlung"""
        error_count = 0
        while self.running:
            if self.cap2 and self.cap2.isOpened():
                ret, frame = self.cap2.read()
                if ret and frame is not None:
                    self.frame2 = frame.copy()
                    error_count = 0  # Reset error counter
                else:
                    error_count += 1
                    if error_count % 30 == 0:  # Alle 30 Fehler melden
                        print(f"Kamera 2: {error_count} Frames verloren")
            time.sleep(0.033)  # ~30fps
    
    def start_capture(self):
        """Startet Capture mit ausführlichem Debug"""
        if not self.initialize_cameras():
            print("\n❌ FEHLER: Kamera-Initialisierung fehlgeschlagen!")
            input("Drücke Enter zum Beenden...")
            return False
        
        self.running = True
        
        print("\nStarte Capture-Threads...")
        self.thread1 = threading.Thread(target=self.capture_thread_cam1, daemon=True)
        self.thread2 = threading.Thread(target=self.capture_thread_cam2, daemon=True)
        
        self.thread1.start()
        self.thread2.start()
        
        print("✓ Threads gestartet")
        
        # Kurz warten und ersten Frame-Test
        time.sleep(1.0)
        
        if self.frame1 is None or self.frame2 is None:
            print("❌ WARNUNG: Keine Frames nach 1 Sekunde!")
            print(f"Frame1: {self.frame1 is not None}, Frame2: {self.frame2 is not None}")
        else:
            print("✓ Erste Frames empfangen")
        
        return True
    
    def run_with_debug(self):
        """Hauptschleife mit Debug-Informationen"""
        if not self.start_capture():
            return
        
        print("\n=== Live-Ansicht gestartet ===")
        print("Tastatur-Shortcuts:")
        print("  'q' - Beenden")
        print("  's' - Screenshot")
        print("  'i' - Info")
        print("  ESC - Beenden")
        
        frame_count = 0
        last_info_time = time.time()
        
        try:
            while True:
                frame1, frame2 = self.frame1, self.frame2
                
                if frame1 is not None and frame2 is not None:
                    # Display-Frames (verkleinert)
                    display1 = cv2.resize(frame1, (640, 360))
                    display2 = cv2.resize(frame2, (640, 360))
                    
                    # Status-Info hinzufügen
                    cv2.putText(display1, f"Cam1: {frame1.shape[1]}x{frame1.shape[0]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(display2, f"Cam2: {frame2.shape[1]}x{frame2.shape[0]}", 
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Anzeigen
                    cv2.imshow('Debug Cam1', display1)
                    cv2.imshow('Debug Cam2', display2)
                    
                    # Combined
                    combined = np.hstack((display1, display2))
                    cv2.putText(combined, f"Frame: {frame_count}", 
                               (combined.shape[1]//2 - 50, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    cv2.imshow('Debug Combined', combined)
                    
                    frame_count += 1
                    
                    # Periodische Info
                    if time.time() - last_info_time > 5.0:
                        print(f"Status: {frame_count} Frames empfangen")
                        last_info_time = time.time()
                
                else:
                    print(f"Frame-Status: Cam1={frame1 is not None}, Cam2={frame2 is not None}")
                
                # Tastaturabfrage
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # 'q' oder ESC
                    break
                elif key == ord('s'):
                    self.save_debug_frames()
                elif key == ord('i'):
                    self.print_debug_info()
        
        except KeyboardInterrupt:
            print("\n⚠️ Unterbrochen durch Benutzer")
        
        except Exception as e:
            print(f"\n❌ Fehler in Hauptschleife: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def save_debug_frames(self):
        """Speichert Frames für Debug-Zwecke"""
        if self.frame1 is not None and self.frame2 is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            cv2.imwrite(f"debug_cam1_{timestamp}.png", self.frame1)
            cv2.imwrite(f"debug_cam2_{timestamp}.png", self.frame2)
            print(f"Debug-Frames gespeichert: debug_cam1/2_{timestamp}.png")
        else:
            print("Keine Frames zum Speichern verfügbar!")
    
    def print_debug_info(self):
        """Druckt Debug-Informationen"""
        print("\n=== DEBUG INFO ===")
        print(f"Running: {self.running}")
        print(f"Frame1 verfügbar: {self.frame1 is not None}")
        print(f"Frame2 verfügbar: {self.frame2 is not None}")
        
        if self.cap1:
            w1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
            h1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps1 = self.cap1.get(cv2.CAP_PROP_FPS)
            print(f"Kamera 1: {w1}x{h1} @ {fps1}fps")
        
        if self.cap2:
            w2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
            h2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps2 = self.cap2.get(cv2.CAP_PROP_FPS)
            print(f"Kamera 2: {w2}x{h2} @ {fps2}fps")
    
    def cleanup(self):
        """Bereinigt alle Ressourcen"""
        print("\n=== Cleanup ===")
        self.running = False
        
        time.sleep(0.5)  # Threads Zeit zum Beenden geben
        
        if self.cap1:
            self.cap1.release()
            print("✓ Kamera 1 freigegeben")
        
        if self.cap2:
            self.cap2.release()
            print("✓ Kamera 2 freigegeben")
        
        cv2.destroyAllWindows()
        print("✓ Alle Fenster geschlossen")

def main():
    """Hauptfunktion mit umfassender Diagnose"""
    print("=== Logitech C310 Debug Version ===")
    print("Diese Version hilft bei der Fehlerdiagnose\n")
    
    try:
        # Umfassender Kamera-Scan
        working_cameras = comprehensive_camera_scan()
        
        if len(working_cameras) < 2:
            print(f"\n❌ FEHLER: Nur {len(working_cameras)} Kamera(s) gefunden!")
            print("Für Dual-Capture werden 2 Kameras benötigt.")
            
            if len(working_cameras) == 1:
                print("\nTeste einzelne Kamera...")
                idx, backend = working_cameras[0]
                cap = cv2.VideoCapture(idx, backend)
                if cap.isOpened():
                    ret, frame = cap.read()
                    if ret:
                        cv2.imshow('Einzelne Kamera Test', cv2.resize(frame, (640, 480)))
                        print("Drücke eine Taste zum Beenden...")
                        cv2.waitKey(0)
                        cv2.destroyAllWindows()
                cap.release()
            
            input("\nDrücke Enter zum Beenden...")
            return
        
        print(f"\n✓ {len(working_cameras)} Kameras gefunden!")
        
        # Erste zwei Kameras für Dual-Capture verwenden
        cam1_config = working_cameras[0]
        cam2_config = working_cameras[1]
        
        print(f"\nVerwende für Dual-Capture:")
        print(f"  Kamera 1: Index {cam1_config[0]}")
        print(f"  Kamera 2: Index {cam2_config[0]}")
        
        # Dual Capture starten
        dual_cam = DebugDualCapture(cam1_config, cam2_config)
        dual_cam.run_with_debug()
        
    except Exception as e:
        print(f"\n❌ Unerwarteter Fehler: {e}")
        import traceback
        traceback.print_exc()
        input("\nDrücke Enter zum Beenden...")

if __name__ == "__main__":
    main()
