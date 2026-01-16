# Token Information Dynamics Control (Early-Exit) for llama.cpp

Dieses Feature implementiert eine dynamische Inferenz-Steuerung basierend auf der **Information Dynamics** der Logits. Es ermöglicht dem Modell, die Generierung vorzeitig abzubrechen, sobald eine statistische Konvergenz oder eine übermäßige energetische Dissipation festgestellt wird.

## Kernkonzepte

### 1. Local Entropy Stability

Anstatt nur auf das `<|end_of_text|>` Token zu warten, misst dieses Feature den **Logit-Gap** (den Abstand zwischen dem wahrscheinlichsten und dem zweitwahrscheinlichsten Token). Ein hoher -Wert signalisiert eine hohe Vorhersage-Sicherheit (**Sense-Prediction**).

### 2. Cross-Layer Refinement Rate & EMA

Die Veränderung der Stabilität über die Zeit wird mittels eines **Exponential Moving Average (EMA)** geglättet. Dies filtert mathematisches Rauschen (besonders bei quantisierten Modellen) heraus und isoliert den tatsächlichen Trend der Informationsdichte.

### 3. Energy Accumulation

Jeder signifikante Wechsel in der Vorhersage-Richtung kostet "Energie". Übersteigt die akkumulierte Energie einen Schwellenwert (**Burnout**), wird die Inferenz beendet. Dies verhindert, dass das Modell in endlose, redundante Schleifen abdriftet.

---

## Vorteile im Vergleich (Beispielszenario)

*Basierend auf einem Test-Setup: Mistral 7B, Q8_0, ~1000 Prompt-Tokens.*

| Merkmal | Standard-Inferenz | Early-Exit |
| --- | --- | --- |
| **Terminierung** | EOS oder Token-Limit | C-Gap oder S-budget-Limit |
| **Effizienz** | 100% Rechenlast | **~30% - 50% Rechenlast** |
| **Verhalten** | Neigt zu Redundanz ("Yapping") | Prägnant, endet am Informations-Peak |
| **Halluzination** | Risiko steigt bei langem Output | **Minimiert** durch rechtzeitigen Abbruch |

### Die Kontrakarrikatur zum Reasoning

Während **Reasoning-Verfahren** (wie Chain-of-Thought) die Rechenzeit durch *Expansion* erhöhen, um Qualität zu erzwingen, ist der **Early-Exit** das reduktive Gegenstück. Er bewahrt die Qualität durch *Selektion*, indem er die Inferenz genau dann stoppt, bevor das statistische Rauschen überhandnimmt.

---

## Anwendung

### Parameter

* `--early-exit`: Aktiviert das Feature. Und deaktiviert mögliches reasoning.
* `--early-exit-gap N`: Schwellenwert für die Stabilität Sprünge. (Default: `6.0`). Höhere Werte führen zu späterem Abbruch bei hoher Sicherheit.
* `--early-exit-burnout N`: Limit für die akkumulierte Energie. (Default: `100.0`). Schützt vor "Drifting" und Endlosschleifen.

### Beispiel-Befehl

```bash
./llama-server -m model.gguf --early-exit --early-exit-gap 12.0 --early-exit-burnout 200.0

```

### Visuelles Feedback

Bei einem vorzeitigen Abbruch injiziert das System ein `...` in den Stream. Dies signalisiert dem Nutzer (oder dem Client), dass die Generierung kontrolliert durch die Dynamics-Aufsicht beendet wurde und keine technische Störung vorliegt.

---

## Technische Implementierung

Die Prüfung erfolgt unmittelbar nach der Logit-Generierung und **vor** der Sampler-Chain. Dadurch wird sichergestellt, dass die Messung auf den unverfälschten Wahrscheinlichkeiten des Modells basiert und Rechenleistung für nicht benötigte Sampling-Schritte eingespart wird.
