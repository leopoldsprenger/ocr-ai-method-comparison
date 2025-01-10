# OCR AI Methodenvergleich

Dieses Repository enthält den Code, der verwendet wurde, um zwei verschiedene Methoden der künstlichen Intelligenz (KI) für die optische Zeichenerkennung (OCR) zu vergleichen. Der Vergleich war Teil meiner wissenschaftlichen Arbeit für die 9. Klasse.

Die beiden verglichenen KI-Methoden sind:
1. **Genetische Algorithmus-basierte KI**: Ein Modell, das mithilfe genetischer Algorithmen optimiert wird.
2. **Gradientenabstieg-basierte KI**: Ein traditionelles neuronales Netzwerk, das mit dem Gradientenabstiegs-Optimierer trainiert wird.

Dieses Projekt untersucht, wie jede Methode bei der Erkennung von handgeschriebenen Ziffern aus dem MNIST-Datensatz abschneidet und vergleicht dabei den Trainingsfortschritt, die Genauigkeit und die Leistung.

---

## Inhaltsverzeichnis
1. [Überblick](#überblick)
2. [Setup und Installation](#setup-und-installation)
3. [Code-Struktur](#code-struktur)
4. [Modell trainieren](#modell-trainieren)
5. [Modell testen](#modell-testen)
6. [Ergebnisse und Vergleich](#ergebnisse-und-vergleich)
7. [Fazit](#fazit)

---

## Überblick

Dieses Projekt zielt darauf ab, zwei verschiedene KI-Trainingsmethoden für OCR-Aufgaben zu vergleichen. Der Vergleich konzentriert sich auf den Trainingsprozess, die Genauigkeit und die Generalisierungsfähigkeit auf dem MNIST-Datensatz, der aus Bildern handgeschriebener Ziffern (0–9) besteht.

## Code-Struktur

### `gradient_descent.py`

Dieses Skript enthält den Code zum Trainieren eines neuronalen Netzwerks mit Gradientenabstieg. Es umfasst die folgenden Hauptkomponenten:

- **Neuronen-Netzwerk-Architektur**: Das Netzwerk besteht aus drei voll verbundenen Schichten mit ReLU-Aktivierung zwischen den Schichten.

- **Trainingsschleife**: Implementiert den Trainingsprozess mit dem Stochastischen Gradientenabstieg (SGD) Optimierer und der Kreuzentropie-Verlustfunktion.

- **Testfunktion**: Bewertet das Modell auf dem Testdatensatz und visualisiert die Vorhersagen.

- **Trainingsvisualisierung**: Plottet den Verlust über die Epochen, um den Fortschritt des Modells zu verfolgen.

### `data_manager.py`

Diese Datei übernimmt das Laden und Verarbeiten des MNIST-Datensatzes. Sie definiert die folgenden Hauptkomponenten:

- **DataLoader**: Eine benutzerdefinierte DataLoader-Klasse, die Batches von Trainingsdaten lädt.

- **Modell speichern und laden**: Funktionen zum Speichern und Laden der Modellgewichte.

## Modell trainieren

Die Funktion `train_model_from_scratch()` in `gradient_descent.py` wird verwendet, um das Modell von Grund auf neu zu trainieren. Sie nutzt den Trainingsdatensatz, um das Modell zu optimieren, und verfolgt den Verlust und die Genauigkeit während der Epochen.

Du kannst das Training starten, indem du das Skript ausführst und den entsprechenden Modus wählst:

1. **Modell von Grund auf neu trainieren**: Trainiert ein neues Modell mit den MNIST-Trainingsdaten und speichert die trainierten Gewichte.
2. **Vorhandenes Modell laden und testen**: Lädt vortrainierte Gewichte und testet das Modell auf dem Testdatensatz.

Führe das Skript wie folgt aus:

```bash
python gradient_descent.py
```

## Modell testen

Nach dem Training wird das Modell mit dem Testdatensatz getestet. Die Funktion `test_model()` bewertet die Leistung und visualisiert die Vorhersagen für die ersten 40 Testbilder.

Die visuelle Ausgabe zeigt die vorhergesagten Ziffern zusammen mit den Eingabebildern, sodass du prüfen kannst, wie gut das Modell abschneidet.

## Ergebnisse und Vergleich

Die Leistung beider Modelle (genetischer Algorithmus und Gradientenabstieg) wird anhand der folgenden Metriken verglichen:

- **Trainingsverlust**: Ein Diagramm des Verlustes über die Epochen für jede Methode.
- **Testgenauigkeit**: Die Genauigkeit auf dem Testdatensatz.

## Fazit

Dieser Vergleich gibt Aufschluss über die Effektivität der verschiedenen KI-Trainingsmethoden für OCR-Aufgaben. Die endgültige Modellleistung wird helfen, zu bestimmen, welche Methode bessere Ergebnisse bei der Erkennung handgeschriebener Ziffern liefert.

## Beispielcode zum Laden und Testen des Modells

Nachdem du das Modell trainiert oder ein vortrainiertes Modell geladen hast, kannst du es mit folgendem Beispiel testen:

```python
neural_network = load_model("model_weights/gradient_descent.pt", NeuralNetwork())
test_model(neural_network)
```

### Bilder und Diagramme

Füge hier Bilder und Diagramme hinzu, die während des Trainingsprozesses generiert wurden. Diese visuellen Hilfsmittel tragen dazu bei, ein besseres Verständnis der Ergebnisse zu vermitteln.

Beispiel eines Diagramms:

```markdown
![Trainingsverlust](images/training_loss.png)
```
