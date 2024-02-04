# Licpl-Censor
Licpl-Censor ist ein Tool, das darauf ausgelegt ist, Gesichter und Kennzeichen in Videos zu zensieren. Es befindet sich derzeit in aktiver Entwicklung, daher sind häufige Updates und potenzielle Fehler zu erwarten.

## Einschränkungen
Das Modell ist speziell auf folgendes trainiert:
- Deutsche Kennzeichen. Es könnte mit anderen EU-Kennzeichen funktionieren, die Ergebnisse können jedoch variieren.
- Dashcam-Videos. Das Modell ist an die Perspektive und Qualität von Dashcam-Videos gewöhnt, daher kann die Leistung bei anderen Videoarten unterschiedlich sein.

Das Datenset wird kontinuierlich aktualisiert und verbessert. Nicht alle Kameratypen werden jedoch bereits gut unterstützt. Da es sich um ein Hobbyprojekt handelt, ist der Prozess der Beschriftung und Verbesserung des Datensatzes zeitaufwendig.

Bitte beachten Sie, dass das Modell möglicherweise Instanzen übersehen kann, was eine manuelle Überprüfung der Ausgabe erforderlich macht. Dennoch reduziert es die Zeit erheblich im Vergleich zur manuellen Zensur.

## Installation

### Eigenständige ausführbare Datei (nur Windows)
Für Windows ist eine eigenständige ausführbare Datei verfügbar.
Es umfasst alle erforderlichen Abhängigkeiten und NVIDIA CUDA-Unterstützung.
Möglicherweise müssen Sie [ffmpeg](https://community.chocolatey.org/packages/ffmpeg) manuell installieren, wenn es noch nicht auf Ihrem System installiert ist oder das Tool es nicht finden kann.

Die eigenständige ausführbare Datei ist in den Github Releases *nicht* verfügbar, aus dem einfachen Grund, dass sie zu groß ist, um sie auf GitHub hochzuladen.
Download-Links sind den Release Notes beigefügt.

Es gibt eine CLI- und eine GUI-Version. Die CLI-Version verfügt über alle Funktionen, während die GUI-Version benutzerfreundlicher ist.

__Hinweis:__ Beim Öffnen der eigenständigen ausführbaren Datei werden die Dateien in ein temporäres Verzeichnis extrahiert. Dies kann je nach System einige Sekunden oder sogar Minuten dauern.

### Docker


1. Klone das Repository.
2. Führe `docker build -t licpl-censor .` im Stammverzeichnis aus.
3. Starte den Container mit `docker run --rm -it -v /Pfad/zu/deinem/Arbeitsverzeichnis:/app/data licpl-censor bash`. Stelle sicher, dass du eventuelle NVIDIA-GPUs weitergibtst, um schnelle Leistung zu gewährleisten.

Du kannst das Tool dann mit folgendem Befehl ausführen:

```bash
python src/main.py --input /app/data/deine-eingabe/ --output /app/data/deine-ausgabe/ --model /app/data/model.pt
```

Das Tool wird versuchen, die Ordnerstruktur deines Eingabeordners im Ausgabeordner zu replizieren. Wenn dies nicht gewünscht ist oder fehlschlägt, kannst du die `--flat-output` Flag verwenden, um dieses Verhalten zu deaktivieren.

Um alle verfügbaren Optionen anzuzeigen, führe `python src/main.py --help` aus.

## Modelbeschaffung
Neue Modelle werden veröffentlicht, sobald sie verfügbar sind. Diese "kleinen" Modelle bieten eine gute Balance zwischen Geschwindigkeit und Genauigkeit. Größere Modelle sind ebenfalls in Entwicklung, stehen aber derzeit nicht zur Verfügung.

# Spenden
Dieses Projekt ist ein Hobby, entwickelt in meiner Freizeit. Wenn du das Wachstum und die Verbesserung des Datensatzes und des Modells unterstützen möchtest, kannst du dies hier tun:

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/clemense)

### Anerkennungen
Das Modell wird mit Ultralytics YOLOv8 trainiert.