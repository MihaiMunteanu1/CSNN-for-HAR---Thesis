
# Recunoașterea Acțiunilor Umane prin Rețele Neurale Spiking Convoluționale pe Datasetul KTH

---

## Introducere și Motivație

### Contextul Problemei: Recunoașterea Automată a Acțiunilor Umane

Recunoașterea automată a acțiunilor umane din secvențe video (**Human Action Recognition — HAR**) reprezintă una dintre problemele fundamentale ale viziunii artificiale, cu aplicații directe în supraveghere inteligentă, interacțiunea om-robot, conducerea autonomă, analiza sportivă și asistența medicală [1, 2]. Problema constă în a asocia unei secvențe de cadre video o etichetă semantică care descrie acțiunea efectuată de persoana observată (ex. „mers", „alergare", „aplauze").

Dificultatea HAR provine din **variabilitatea masivă** a reprezentării vizuale a aceleiași acțiuni: diferențe de iluminare, unghi de filmare, distanță față de cameră, îmbrăcăminte, morfologie corporală, viteză de execuție și fundal. O soluție robustă trebuie să fie **invariantă** la aceste variații irelevante, extrăgând doar esența cinematică a mișcării.

### De ce Rețele Neurale Spiking?

Abordările dominante în HAR se bazează pe **rețele neurale convoluționale profunde (CNN)** — arhitecturi precum C3D [3], Two-Stream Networks [4], I3D [5] sau SlowFast [6] — care ating acuratețe de peste 95% pe benchmark-uri standard. Însă aceste rețele operează cu **activări continue** ($a \in \mathbb{R}$) propagate sincron prin zeci sau sute de straturi, necesitând:
- **Putere de calcul intensivă**: miliarde de operații în virgulă mobilă (FLOPs) per predicție
- **Consum energetic ridicat**: GPU-urile moderne consumă 200–400W în timpul inferenței
- **Hardware specializat**: procesarea nu poate fi eficientizată pe dispozitive edge cu buget energetic limitat

**Rețelele Neurale Spiking (SNN — Spiking Neural Networks)** [7, 8] propun o paradigmă alternativă, inspirată direct din modul în care **cortexul vizual al creierului** procesează informația:

1. **Comunicare prin spike-uri binare**: Neuronii nu transmit valori continue, ci impulsuri discrete (totul-sau-nimic), similare cu potențialele de acțiune din neuronii biologici. Consecința: fiecare sinapsă efectuează o **adunare**, nu o multiplicare — reducând dramatic complexitatea computațională.

2. **Codificare temporală**: Informația este codificată în **momentul** emiterii spike-ului, nu în amplitudine. Un stimul puternic generează un spike devreme ($t_s \approx 0$), unul slab — târziu ($t_s \approx 1$). Această codificare, numită **rank-order coding** [9], permite procesarea cu un singur pas temporal, nu cu sute de epoci.

3. **Procesare event-driven**: Calculul are loc doar acolo unde există activitate (spike-uri). Zonele statice (fundal) nu consumă energie. Estimările arată că SNN-urile sunt teoretic de **10–1000× mai eficiente energetic** decât CNN-urile echivalente [10], deschizând calea spre:
   - **Hardware neuromorfic**: cipuri precum Intel Loihi [11], IBM TrueNorth [12] și SpiNNaker [13] sunt proiectate nativ pentru procesarea spike-urilor, consumând sub 1W
   - **Dispozitive edge**: camere de supraveghere, drone, roboți autonomi cu procesare on-board
   - **Camere eveniment (DVS)**: senzori care emit spike-uri direct, fără cadre intermediare [14]

### Învățarea Nesupervizată prin STDP

O a doua motivație fundamentală este **absența necesității etichetelor** de antrenare. CNN-urile supervizate necesită mii sau milioane de imagini etichetate manual — un proces costisitor și neescalabil. SNN-urile din prezenta lucrare învață prin **STDP (Spike-Timing-Dependent Plasticity)** [15, 16], o regulă de plasticitate sinaptică descoperită experimental în neuroni biologici:

> *Dacă neuronul A emite un spike imediat înainte ca neuronul B să tragă, sinapsa $A \to B$ este întărită (neuronul A a contribuit cauzal la activarea lui B). Dacă ordinea este inversată, sinapsa este slăbită.*

Această regulă simplistă, aplicată milioane de ori peste datele de antrenare, auto-organizează filtrele convoluționale pentru a detecta **structuri statistice recurente** — margini, texturi, direcții de mișcare — fără supervizare explicită. Filtrele emergente sunt remarcabil de similare cu **celulele simple** din cortexul vizual primar V1 [17], confirmând plauzibilitatea biologică a abordării.

### Obiectivul Lucrării

Prezenta lucrare propune și implementează un **pipeline complet de recunoaștere a acțiunilor umane** pe datasetul KTH [18], combinând:

1. **Preprocesare inteligentă** cu detecție duală HOG+MOG2 [19, 20] pentru localizarea persoanei în cadru
2. **O rețea neurală spiking convoluțională (CSNN)** cu un strat convoluțional 3D antrenat nesupervizat prin STDP Biologic [16]
3. **Eșantionare ghidată (HOGSampler3D)** — un mecanism original de focalizare a atenției spațiale pe baza bounding box-urilor pre-calculate
4. **Clasificare supervizată SVM** [21] pe reprezentările spike extrase de CSNN

Arhitectura folosită combină o rețea neurală spiking convoluțională (**CSNN - Convolutional Spiking Neural Network**) capabilă să asimileze vizual secvențe spatio-temporale nesupervizate, stabilizată de o tehnică de detecție biologică HOG pentru direcționarea atenției spațiale a modelului (Region of Interest) și un clasificator tradițional (SVM) pentru validarea supervizată a caracteristicilor vizuale obținute.

Spre deosebire de arhitecturile clasice de Deep Learning (tip CNN / ResNet) care procesează imagini dense matematic cu un consum energetic ridicat, abordarea curentă folosește **Paradigma Neuromorfică**. Aceasta traduce informația vizuală în impulsuri binare asincrone (Spike-uri), procesând strict evenimentele active ale mișcării umane, mimând astfel eficiența energetică și topologia creierului uman.

**[* FIGURA INTRO: O diagramă conceptuală cu 3 coloane: (1) Creierul biologic (neuron, sinapsă, spike), (2) SNN-ul propus (integrate-and-fire, STDP, convoluție spiking), (3) CNN clasic (ReLU, backpropagation, convoluție densă). Săgeți indicând inspirația biologică de la (1) la (2) și contrastul computațional între (2) și (3). *]**

---

## Stadiul Actual al Cercetării (Related Work)

### Recunoașterea Acțiunilor Umane: De la Trăsături Manuale la Deep Learning

Cercetarea în HAR a evoluat prin trei paradigme succesive:

#### Faza 1: Trăsături Manuale (Hand-Crafted Features) — 2000–2012

Primele abordări se bazau pe **descriptori proiectați manual** de cercetători, extrași din video și clasificați cu SVM sau Random Forest:

- **Space-Time Interest Points (STIP)** [22]: Laptev (2005) extinde detectorul de colțuri Harris în domeniul spatio-temporal, identificând puncte unde există variație simultană în spațiu și timp. Descriptorul combină HOG (gradient spațial) cu HOF (gradient de flux optic).

- **Dense Trajectories** [23]: Wang et al. (2011) urmăresc puncte dense prin flux optic pe mai multe cadre, extrăgând descriptori HOG, HOF și MBH (Motion Boundary Histograms) de-a lungul traiectoriilor. Această metodă a dominat benchmark-urile HAR până la apariția deep learning-ului.

- **Bag of Visual Words (BoVW)** [24]: Trăsăturile locale sunt cuantizate într-un vocabular fix prin k-means, iar fiecare video este reprezentat ca o histogramă de frecvențe ale cuvintelor vizuale.

Pe datasetul KTH, aceste abordări atingeau **85–92%** acuratețe [18, 22, 23].

#### Faza 2: Rețele Convoluționale 3D (CNN-uri spatio-temporale) — 2014–2020

Apariția GPU-urilor performante și a dataset-urilor mari (UCF-101, Kinetics) a permis antrenarea rețelelor profunde direct pe secvențe video:

- **Two-Stream Networks** [4]: Simonyan & Zisserman (2014) propun două rețele CNN paralele — una pe cadre RGB (aparență) și una pe flux optic (mișcare) — ale căror predicții sunt fuzionate. Acuratețe pe KTH: **93–95%**.

- **C3D (Convolutional 3D)** [3]: Tran et al. (2015) extind convoluțiile 2D la 3D ($3 \times 3 \times 3$), procesând simultan informația spațială și temporală. 8 straturi convoluționale cu milioane de parametri antrenați supervizat pe Sports-1M. Acuratețe pe KTH: **90–93%**.

- **I3D (Inflated 3D)** [5]: Carreira & Zisserman (2017) „inflatează" rețele 2D pre-antrenate pe ImageNet (ex. Inception) în 3D, transferând cunoștințele din domeniul imaginilor la cel al videourilor. Acuratețe pe KTH: **95–98%**.

- **SlowFast Networks** [6]: Feichtenhofer et al. (2019) folosesc două ramuri cu rezoluții temporale diferite — una lentă (semantică) și una rapidă (mișcare) — obținând rezultate state-of-the-art pe Kinetics.

Aceste arhitecturi, deși extrem de precise, necesită **milioane de parametri**, antrenare supervizată pe sute de mii de videoclipuri etichetate, și hardware GPU costisitor.

#### Faza 3: Abordări Neuromorfice și SNN — 2017–prezent

Cercetarea recentă explorează utilizarea SNN-urilor pentru sarcinile de viziune artificială, inclusiv HAR:

- **Kheradpisheh et al. (2018)** [25]: Propun o arhitectură CSNN cu STDP nesupervizat pentru recunoașterea obiectelor statice (MNIST, Caltech, ETH-80). Demonstrează că STDP-ul auto-organizează filtre similare cu celulele simple din cortexul V1. Acuratețe pe MNIST: 98.4%.

- **Mozafari et al. (2019)** [26]: Extind abordarea CSNN cu un mecanism de **Reward-Modulated STDP (R-STDP)**, adăugând un semnal de recompensă care ghidează STDP-ul spre filtre discriminative. Acuratețe pe MNIST: 97.2% (nesupervizat) și 98.5% (R-STDP).

- **Falez et al. (2019)** [27]: Simulatorul CSNN pe care se bazează prezenta lucrare. Propune o arhitectură convoluțională spiking cu antrenare STDP, mecanisme Winner-Takes-All, și clasificare SVM. Evaluat pe MNIST și N-MNIST.

- **Parameshwara et al. (2021)** [28]: Aplică SNN-uri pe date de la camere eveniment (DVS) pentru recunoașterea gesturilor și acțiunilor. Demonstrează avantajul nativ al procesării event-driven pentru senzori care emit deja spike-uri.

- **Fang et al. (2021)** [29]: Propun SpikingJelly, un framework pentru SNN-uri profunde, și demonstrează că SNN-urile pot atinge performanțe comparabile cu ANN-urile pe ImageNet când sunt antrenate cu surrogate gradient.

### Detectorul HOG: Fundament și Evoluție

**Histogram of Oriented Gradients (HOG)** [19] a fost propus de Dalal & Triggs (2005) pentru detecția pietonilor. Algoritmul:
1. Calculează gradientul de intensitate la fiecare pixel: magnitudine și orientare
2. Divide imaginea în celule de $8 \times 8$ pixeli
3. Construiește histograme de orientare (9 bin-uri pe $0°$–$180°$) per celulă
4. Normalizează histogramele în blocuri de $2 \times 2$ celule pentru invarianță la iluminare
5. Concatenează toate histogramele normalizate într-un descriptor global
6. Antrenează un clasificator **SVM liniar** pe descriptori HOG pozitivi (persoane) și negativi (fundal)

HOG rămâne relevant în 2024+ pentru detecția pietonilor pe hardware limitat datorită simplității sale computaționale și lipsei necesității de GPU. În prezenta lucrare, HOG este utilizat **offline** pentru localizarea persoanei în etapa de preprocessare, nu la runtime.

### Background Subtraction: MOG2

**Mixture of Gaussians v2 (MOG2)** [20] este un algoritm de segmentare a fundalului propus de Zivkovic (2004, 2006). Pentru fiecare pixel, modelează distribuția de intensitate ca o **mixtură de gaussiene** adaptivă:

$$p(x_t) = \sum_{k=1}^{K} \omega_k \cdot \mathcal{N}(x_t \mid \mu_k, \sigma_k^2)$$

unde $K$ (numărul de componente) este selectat automat per pixel (3–5 componente tipic). Pixelii a căror intensitate curentă nu se potrivește cu nicio componentă sunt clasificați ca **foreground** (mișcare). Avantajul MOG2 față de MOG1: selecția automată a lui $K$ și adaptarea mai rapidă la schimbări de iluminare.

### Clasificatorul SVM (Support Vector Machine)

**Support Vector Machine** [21] este un clasificator supervizat care găsește **hiperplanul de separare optimă** între două clase, maximizând marginea (distanța minimă de la hiperplan la cele mai apropiate puncte — vectorii suport):

$$\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i (w^T x_i + b) \geq 1, \; \forall i$$

Pentru probleme multi-clasă (6 acțiuni KTH), se utilizează strategia **one-vs-one** (15 clasificatoare binare, cu vot majoritar) sau **one-vs-all** (6 clasificatoare). În prezenta lucrare, libsvm [30] implementează varianta one-vs-one cu kernel liniar.

### Poziționarea Prezentei Lucrări

Tabelul următor situează abordarea propusă în contextul literaturii:

| Abordare | Tip învățare | Intrare | Straturi | Parametri | KTH Acc. |
|----------|-------------|---------|----------|-----------|----------|
| STIP + BoVW [22] | — (hand-crafted) | Video | — | — | ~85% |
| Dense Trajectories [23] | — (hand-crafted) | Video + OF | — | — | ~92% |
| Two-Stream CNN [4] | Supervizat | RGB + OF | ~16+16 | ~24M | ~94% |
| C3D [3] | Supervizat | Video | 8 conv + 2 FC | ~78M | ~91% |
| I3D [5] | Supervizat | RGB + OF | ~50+ | ~25M | ~97% |
| Kheradpisheh CSNN [25] | STDP (nesupervizat) | Imagine | 3 conv | ~100K | N/A (imagini) |
| Mozafari R-STDP [26] | R-STDP (semi) | Imagine | 2 conv | ~50K | N/A (imagini) |
| **Prezenta lucrare** | **STDP (nesupervizat)** | **Video 3D** | **1 conv** | **~6.4K** | **[* TBD *]** |

Contribuția principală nu este de a *depăși* performanța CNN-urilor supervizate, ci de a demonstra fezabilitatea unei arhitecturi CSNN nesupervizate pe date video 3D reale, cu un pipeline end-to-end care include preprocesare inteligentă (HOG+MOG2), eșantionare ghidată (HOGSampler3D) și extracție de trăsături spatio-temporale prin STDP biologic.

**[* FIGURA RELATED WORK: Un timeline vizual (2005–2024) cu milestone-urile HAR: HOG (2005) → STIP (2005) → Dense Traj. (2011) → Two-Stream (2014) → C3D (2015) → I3D (2017) → SlowFast (2019) → SNN HAR (2021+). Marchează poziția lucrării curente pe timeline. *]**

---

## Fundamentele Rețelelor Neurale Spiking Convoluționale (CSNN)

### Paradigma Neuromorfică vs. Deep Learning Clasic

Rețelele neurale artificiale tradiționale (ANN / CNN) [32] operează cu **valori continue** (activări reale $a \in \mathbb{R}$) propagate sincron prin toate straturile rețelei. Fiecare neuron calculează o sumă ponderată a inputurilor, aplică o funcție de activare (ReLU, Sigmoid), și transmite rezultatul numeric mai departe. Această paradigmă, antrenată prin backpropagation [31], implică un consum computațional ridicat — fiecare pixel, fiecare neuron, la fiecare pas temporal, contribuie la calcul.

**Rețelele Neurale Spiking (SNN - Spiking Neural Networks)** [7, 8] adoptă un model fundamental diferit, inspirat direct din neuroștiință:
- **Comunicarea prin spike-uri**: Neuronii nu transmit valori continue, ci **impulsuri binare discrete** (spike-uri / acțiuni de potențial). Un neuron fie „trage" (emite spike), fie tace — nu există valori intermediare.
- **Codificarea temporală**: Informația nu este codificată în *amplitudinea* activării, ci în **momentul temporal** al spike-ului. Un pixel luminos (important) emite un spike *devreme* ($t_s \approx 0$), iar un pixel întunecat (irelevant) emite *târziu* ($t_s \approx 1$) sau deloc.
- **Procesarea asincronă**: Spike-urile sunt procesate **pe măsură ce sosesc** (event-driven), nu în batch-uri sincrone. Zonele fără activitate (fundal static) nu consumă calcul.

### Modelul Neuronului Spiking (Integrate-and-Fire)

Fiecare neuron din rețea menține un **potențial de membrană** $V(t)$ care se acumulează pe măsură ce primește spike-uri de intrare:

$$V_j(t) = V_j(t-1) + \sum_{i=1}^{N} w_{ij} \cdot s_i(t)$$

unde:
- $V_j(t)$ = potențialul de membrană al neuronului $j$ la momentul $t$
- $w_{ij}$ = ponderea sinaptică dintre neuronul de intrare $i$ și neuronul $j$
- $s_i(t) \in \{0, 1\}$ = spike-ul de la neuronul $i$ la momentul $t$ (1 = spike emis, 0 = tăcere)
- $N$ = numărul total de sinapse de intrare

Când potențialul depășește **pragul de descărcare** $\theta_j$, neuronul emite un spike și potențialul este resetat:

$$\text{dacă } V_j(t) \geq \theta_j \implies \text{spike}_j(t) = 1, \quad V_j(t) \leftarrow 0$$

### Convoluția Spiking 3D (Spatio-Temporală)

**CSNN (Convolutional SNN)** extinde modelul SNN cu operatorul de convoluție spatio-temporală, identic structural cu CNN-urile clasice dar operat pe spike-uri:

Un **filtru 3D** de dimensiuni $f_h \times f_w \times f_t$ (înălțime × lățime × adâncime temporală) glisează peste tensorul de input $[H \times W \times C \times T]$, calculând la fiecare poziție $(x, y, k)$ suma ponderată:

$$V_j(x, y, k) = \sum_{t=0}^{f_t-1} \sum_{c=0}^{C-1} \sum_{dy=0}^{f_h-1} \sum_{dx=0}^{f_w-1} w_j(dx, dy, c, t) \cdot s(x+dx,\; y+dy,\; c,\; k+t)$$

Dimensiunile de output se calculează ca:
$$H_{out} = \frac{H - f_h}{stride_y} + 1, \quad W_{out} = \frac{W - f_w}{stride_x} + 1, \quad T_{out} = \frac{T - f_t}{stride_k} + 1$$

Numărul de **sinapse per neuron** (ponderi individuale pe care fiecare filtru le învață) este:
$$N_{sinapse} = f_w \times f_h \times C_{in} \times f_t$$

### Învățarea Nesupervizată: STDP (Spike-Timing-Dependent Plasticity)

Spre deosebire de rețelele tradiționale care folosesc **backpropagation** [31] (propagarea erorilor supervizate), SNN-urile CSNN din prezenta arhitectură învață **nesupervizat** prin **STDP** [15, 16] — o regulă de plasticitate sinaptică descoperită experimental de Bi & Poo (1998) în neuroni biologici.

**Principiul cauzalității**: Dacă un neuron de intrare emite un spike ($t_{pre}$) *înainte* ca neuronul de ieșire să tragă ($t_{post}$), legătura sinaptică a fost **cauzală** (a contribuit la descărcare) și este **întărită** (Long-Term Potentiation — LTP). Dacă ordinea este inversată (post-ul trage înainte de pre), legătura este **slăbită** (Long-Term Depression — LTD):

$$\Delta w_{ij} = \begin{cases} A_+ \cdot e^{-|\Delta t| / \tau_+} & \text{dacă } \Delta t = t_{post} - t_{pre} > 0 \quad \text{(LTP — potențare)} \\ -A_- \cdot e^{-|\Delta t| / \tau_-} & \text{dacă } \Delta t \leq 0 \quad \text{(LTD — depresie)} \end{cases}$$

Prin STDP, filtrele convoluționale se **auto-organizează** pentru a recunoaște pattern-uri recurente din datele de intrare — fără etichete, fără funcție de pierdere (loss function), fără gradient. Masquelier & Thorpe (2007) [17] au demonstrat că STDP-ul auto-organizează filtre similare cu celulele simple descoperite de Hubel & Wiesel (1962) [34] în cortexul vizual V1.

---

## Privire de Ansamblu: Pipeline-ul Complet de la Video Brut la Clasificare

Înainte de a intra în detalii, această secțiune prezintă **fluxul complet** al experimentului KTH_3D, de la videoclipul brut la acuratețea finală:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ETAPA OFFLINE (Python, o singură rulare)            │
│                                                                           │
│  599 videoclipuri KTH (160×120, 25 fps)                                   │
│         │                                                                 │
│        ▼                                                                 │
│  extract_bboxes_kth.py                                                    │
│    • Redimensionare la 80×60                                              │
│    • HOG (upscale 3×→240×180) + MOG2 (background subtraction)             │
│    • Fuziune per-cadru, scorare, EMA smoothing                            │
│    • Selecție 10 grupuri temporale × 5 cadre / video                      │
│         │                                                                 │
│         ▼                                                                 │
│  hog_person_data_5.json  (594/599 videouri, bboxes + frame indices)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ETAPA RUNTIME (C++, simulatorul CSNN)              │
│                                                                           │
│  VideoKTH_3D  ──── citește JSON + video ────►  Tensor [60×80×1×5]         │
│    • 4 sample-uri per video (4 grupuri din 10)                            │
│    • Seek direct la cadrele din JSON                                      │
│    • Construiește mapare sample_index → (video_key, group_idx)            │
│                                                                           │
│         │                                                                 │
│         ▼                                                                 │
│  Preprocesare biologică:                                                  │
│    DefaultOnOffFilter(7, σc=1.0, σs=4.0)  →  [60×80×2×5]  (ON + OFF)     │
│    MaxScaling                             →  [60×80×2×5]  (normalizat)   │
│    LatencyCoding                          →  [60×80×2×5]  (spike times)  │
│                                                                           │
│         │                                                                 │
│         ▼                                                                 │
│  Conv1 (Convolution3D, 64 filtre 5×5×2, HOGSampler3D):                   │
│    ┌───────────────────────────────────────────────────┐                   │
│    │  100 epoci × ~2376 sample-uri:                   │                   │
│    │    1. HOGSampler3D → Patch3D(x,y,k) în bbox      │                   │
│    │    2. Extragere patch 5×5×2×2 (100 sinapse)      │                   │
│    │    3. Integrare spike-uri → potențial membranei   │                   │
│    │    4. Depășire prag → STDP actualizează ponderi   │                   │
│    │    5. WTA + Homeostazie praguri                   │                   │
│    │    6. Annealing lr × 0.95 per epocă              │                   │
│    └───────────────────────────────────────────────────┘                   │
│    Output: [56×76×64×4]                                                   │
│                                                                           │
│         │                                                                 │
│         ▼                                                                 │
│  Postprocessing:                                                          │
│    TimeObjectiveOutput(t_obj=0.75)  →  feature maps sparse               │
│    SumPooling(2×2)                  →  [28×38×64×4]                       │
│    FeatureScaling                   →  normalizat [0,1]                   │
│                                                                           │
│         │                                                                 │
│         ▼                                                                 │
│  Analize:                                                                 │
│    Activity   → sparsity, active units                                    │
│    Coherence  → diversitate filtre (cosine similarity)                    │
│    SVM        → clasificare supervizată pe 6 clase → acuratețe (%)       │
└─────────────────────────────────────────────────────────────────────────────┘
```

**[* FIGURA 0 (IMPORTANTĂ): Redesenează diagrama de mai sus ca o figură grafică profesională (PowerPoint, draw.io, sau LaTeX TikZ), cu blocuri colorate: albastru pentru offline, verde pentru preprocesare, portocaliu pentru antrenare STDP, roșu pentru clasificare SVM. Include dimensiunile tensoriale la fiecare săgeată. *]**

---

## 1. Setul de Date KTH (KTH Video Dataset)

Cunoscut ca unul dintre primele seturi de date standardizate (benchmark) pentru recunoașterea acțiunilor umane, **KTH Video Dataset** [18] (introdus de Schuldt et al., 2004) conține înregistrări video focusate pe mișcări și posturi anatomice distincte. Setul este compus din șase tipuri de acțiuni umane fundamentale:
- **Atingerea adversarului imaginar / Box (Boxing)**
- **Bătutul din palme (Hand Clapping)**
- **Fluturarea mâinilor (Hand Waving)**
- **Mersul normal (Walking)**
- **Alergarea ușoară (Jogging)**
- **Alergarea rapidă (Running)**

Acțiunile sunt interpretate independent de 25 de subiecți (actori) diferiți. Pentru a aduce diversitate și robustețe în generalizarea modelelor vizuale, fiecare secvență este filmată în 4 scenarii izolate vizual: *(s1)* în aer liber (outdoors), *(s2)* în aer liber cu modificarea scalei prin depărtare de cameră, *(s3)* în aer liber cu variații de vestimentație și *(s4)* în mediu controlat în interior (indoors).

**[* FIGURA 1: Un colaj cu 6 cadre reprezentative (screenshot-uri), câte unul pentru fiecare acțiune din dataset-ul KTH, pentru a ilustra formatul vizual al bazei de date. *]**

Materialul brut este captat cu un fundal relativ static și omogen (la frecvența temporală de 25 fps). Dataset-ul conține un total de **599 videoclipuri** distribuite pe cele 6 acțiuni și 4 scenarii. În contextul prezentei arhitecturi CSNN, rezoluția fișierelor originale (160×120 pixeli) este redimensionată la **80×60 pixeli** tocmai pentru a testa eficiența rețelelor Spiking orientate pe contur fluid în detrimentul puterii de calcul exorbitante pe care ar fi necesitat-o o arhitectură clasică prelucrând fiecare pixel la rezoluție mare.

### 1.1. Organizarea Dataset-ului pe Disc și Split-ul Train/Test

Dataset-ul KTH este organizat pe disc într-o structură ierarhică standardizată, pregătită pentru a fi consumată direct de clasa `VideoKTH_3D`:

```
kth_organized/
├── train/
│   ├── boxing/
│   │   ├── person01_boxing_d1_uncomp.avi    (subiect 1, scenariu s1)
│   │   ├── person01_boxing_d2_uncomp.avi    (subiect 1, scenariu s2)
│   │   ├── person01_boxing_d3_uncomp.avi    (subiect 1, scenariu s3)
│   │   ├── person01_boxing_d4_uncomp.avi    (subiect 1, scenariu s4)
│   │   ├── person02_boxing_d1_uncomp.avi
│   │   └── ...
│   ├── handclapping/
│   ├── handwaving/
│   ├── jogging/
│   ├── running/
│   └── walking/
└── test/
    ├── boxing/
    ├── handclapping/
    └── ...
```

**Split-ul standard KTH** separă datele **pe subiecți** (nu pe clipuri individuale), conform protocolului introdus de Schuldt et al. (2004):
- **Train**: Subiecții 1–16 (16 persoane × 6 acțiuni × 4 scenarii = **384 videoclipuri**)
- **Test**: Subiecții 17–25 (9 persoane × 6 acțiuni × 4 scenarii = **~215 videoclipuri**)
- **Total**: 384 + 215 = **599 videoclipuri** (câteva clipuri lipsesc din dataset-ul original)

Separarea pe subiecți este esențială: un model care ar fi antrenat și testat pe aceeași persoană ar putea memora postura specifică a persoanei în loc de a învăța acțiunea generică. Prin split-ul pe subiecți, setul de test conține **persoane complet noi**, forțând modelul să generalizeze dincolo de aparența individuală.

| Split | Subiecți | Videoclipuri | Cu 4 sample-uri/video | Total sample-uri |
|-------|----------|-------------|----------------------|-----------------|
| Train | 1–16     | ~384        | 384 × 4 = 1536      | **~1536**       |
| Test  | 17–25    | ~215        | 215 × 4 = 860       | **~860**        |
| **Total** | 25   | **599**     |                      | **~2396**       |

**[* FIGURA 1b: O captură de ecran a structurii de foldere `kth_organized/` din file manager, arătând ierarhia train/test → acțiuni → videoclipuri. *]**

---

## 2. Preprocesarea Inteligentă a Cadrelor: Scriptul `extract_bboxes_kth.py`

Un element central al pipeline-ului propus este **extragerea offline a cadrelor relevante** din fiecare videoclip KTH, realizată printr-un script Python creat de autor (`src/tool/extract_bboxes_kth.py`). Spre deosebire de abordarea clasică în care cadrele sunt extrase secvențial sau aleatoriu din flux (ceea ce duce frecvent la selectarea unor momente statice sau la pierderea persoanei din cadru), acest script analizează întregul videoclip cadru cu cadru folosind biblioteca OpenCV [38], combinând **doi algoritmi de detecție complementari** — HOG [19] și MOG2 [20] — printr-un mecanism de **fuziune per-cadru** cu netezire temporală, identificând zonele unde persoana este vizibilă și selectând **grupuri temporale optime** de câte **5 cadre**.

### 2.1. Structura unui Grup Temporal

Fiecare grup temporal este format din **5 cadre** (parametrul `temporal_kernel = 5`) dispuse simetric în jurul unui cadru central, cu un **pas temporal** (frame gap) de **2 cadre** intermediare (`frame_gap = 2`):

$$\text{Grup} = [F_{c-4},\; F_{c-2},\; F_{c},\; F_{c+2},\; F_{c+4}]$$

unde $c$ este indexul cadrului central, iar offseturile sunt calculate ca $o \times gap$ pentru $o \in \{-2, -1, 0, +1, +2\}$. Cele 5 cadre ale unui grup acoperă o fereastră temporală de $8 + 1 = 9$ cadre din video (**360 ms** la 25 fps), suficientă pentru a surprinde un ciclu complet de mișcare (extensia și retragerea brațului în boxing, doi pași consecutivi în walking).

Numărul maxim de grupuri extrase per videoclip este configurat la **10 grupuri** (`num_groups = 10`), din care experimentul CSNN folosește **4 sample-uri per video** (`train_sample_per_video = 4`, `test_sample_per_video = 4`), selectând cele mai relevante 4 grupuri din cele 10 disponibile.

### 2.2. Detectorul Primar: HOG (Histogram of Oriented Gradients)

Algoritmul principal de detectare a persoanei este **HOG cu clasificator SVM**, implementat prin OpenCV (`cv2.HOGDescriptor` cu `getDefaultPeopleDetector()`). Acesta funcționează astfel:

1. **Calculul Gradienților Pixelari**: Imaginea este derivată matricial la nivel de pixel pentru a obține magnitudinea și direcția orientării fiecărei margini organice (conturul cămășii, granița corp-fundal).
2. **Construirea Celulelor de Histograme**: Imaginea este segmentată în celule optice ($8 \times 8$ pixeli). Pentru fiecare celulă, vectorii gradient „votează" direcția dominantă în intervalul 0–180°, formând o histogramă locală de orientări.
3. **Normalizarea pe Blocuri**: Celulele sunt agregate în blocuri suprapuse ($16 \times 16$ pixeli), histogramele fiind normalizate L2 pentru insensibilitate la variațiile de iluminare (umbră, reflexii indoor vs. outdoor).
4. **Clasificarea SVM cu Fereastră Glisantă**: Descriptorul HOG vectorial parcurge cadrul prin ferestre glisante (sliding windows) de dimensiuni multiple. Modelul SVM pre-antrenat validează binar dacă conținutul ferestrei corespunde unei siluete umane, returnând coordonatele și scorul de confidență al detecției.

Deoarece rezoluția de lucru (**80×60**) este prea mică pentru detectorul HOG standard (care necesită minimum 64×128 pixeli per fereastră), scriptul **upscalează cadrul cu un factor 3×** (de la 80×60 la **240×180** pixeli) înainte de detecție, apoi scalează bounding box-urile rezultate înapoi la rezoluția originală:

$$sx = \frac{80}{240}, \quad sy = \frac{60}{180}$$
$$x_{orig} = \lfloor x_{det} \times sx \rfloor, \quad y_{orig} = \lfloor y_{det} \times sy \rfloor$$

Parametrul `hit_threshold = -0.75` (negativ) permite un prag mai permisiv de detecție, compensând calitatea redusă a cadrelor KTH. Parametrii ferestrei glisante sunt `winStride = (12, 12)`, `padding = (4, 4)` și `scale = 1.10`.

### 2.3. Detectorul Secundar: MOG2 (Background Subtraction)

Spre deosebire de versiunile anterioare ale pipeline-ului unde MOG2 era folosit doar ca **fallback** (activat numai dacă HOG nu detecta nimic pe întregul videoclip), în versiunea curentă a scriptului, MOG2 rulează **pe fiecare cadru simultan cu HOG**, iar rezultatele celor doi detectori sunt **fuzionate per-cadru** printr-un algoritm de scorare multi-criterială.

MOG2 funcționează pe un principiu fundamental diferit față de HOG:
- **HOG** caută o **formă specifică** (siluetă umană) folosind gradienți și un model pre-antrenat — este un detector de obiect.
- **MOG2** caută **mișcare** detectând pixelii care diferă de fondul static învățat — este un detector de prim-plan (foreground).

Algoritmul MOG2 implementat:
1. **Modelarea Fondului**: Se creează un `BackgroundSubtractorMOG2` cu `history = 120` cadre și `varThreshold = 16`. Fiecare pixel este modelat ca o mixtură de distribuții Gaussiene ce descriu valorile „normale" ale fondului static.
2. **Segmentarea Prim-Planului**: Cadrul curent este comparat cu modelul fondului; pixelii care deviază semnificativ sunt clasificați ca prim-plan (`fgmask`).
3. **Operații Morfologice**: Masca rezultată trece prin operații de **closing** (umplerea golurilor) și **opening** (eliminarea zgomotului) cu un element structural eliptic $5 \times 5$.
4. **Extragerea Bounding Box-ului**: Se găsește cel mai mare contur din masca de prim-plan. Dacă aria sa depășește pragul minim (`mog2_min_area = 180` pixeli pentru rezoluția 80×60), se calculează dreptunghiul înconjurător.

### 2.4. Fuziunea Per-Cadru HOG + MOG2

Elementul distinctiv al versiunii curente a scriptului este **fuziunea inteligentă per-cadru** a detecțiilor HOG și MOG2. Pentru fiecare cadru al videoclipului, ambii detectori produc liste de candidați (bounding box-uri), care sunt apoi evaluate, împerecheate și fuzionate printr-un pipeline în mai mulți pași.

#### 2.4.1. Filtrarea Calității

Înainte de fuziune, fiecare bounding box trece prin două filtre de calitate:

**Clipping** (`clip_bbox`): Coordonatele sunt forțate în limitele cadrului $[0, W) \times [0, H)$.

**Filtrarea de calitate** (`passes_bbox_quality`): Se elimină detecțiile:
- Cu aria sub un prag minim: $w \times h < W \times H \times 0.008$
- Cu aspect ratio (raportul lățime/înălțime) în afara intervalului $[0.22, 1.6]$, eliminând formele exagerat de lungi sau late care nu pot fi siluete umane.

#### 2.4.2. Normalizarea Confidențelor

Confidențele celor doi detectori operează pe scale complet diferite (HOG returnează scoruri SVM, MOG2 returnează aria conturului). Funcția `normalize_confidences` aduce ambele seturi în intervalul $[0, 1]$ prin normalizare min-max:

$$\hat{c}_i = \frac{c_i - c_{min}}{c_{max} - c_{min}}$$

#### 2.4.3. Împerecherea IoU și Fuziunea Geometrică

Funcția centrală `merge_hog_mog2` realizează fuziunea în pași:

1. **Împerecherea prin IoU (Intersection over Union)**: Fiecare detecție HOG este comparată cu fiecare detecție MOG2 prin:
$$IoU(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
Dacă $IoU \geq 0.35$ (pragul `iou_merge_th`), cele două detecții sunt considerate ca referindu-se la aceeași persoană și sunt **fuzionate geometric**:
$$x_{fused} = 0.6 \cdot x_{HOG} + 0.4 \cdot x_{MOG2}$$
$$w_{fused} = 0.5 \cdot w_{HOG} + 0.5 \cdot w_{MOG2}$$

HOG primește ponderi mai mari în **poziție** (0.6 vs 0.4) deoarece detectează silueta exactă, în timp ce dimensiunile sunt mediate egal (0.5/0.5) pentru a beneficia de acoperirea completă a MOG2.

2. **Detecțiile neîmperecheate**: Detecțiile HOG fără corespondent MOG2 și detecțiile MOG2 fără corespondent HOG rămân ca candidați independenți.

#### 2.4.4. Scorarea Multi-Criterială

Din toți candidații (fuzionați + neîmperecheați), se selectează **un singur bbox final** prin funcția `score_box` care combină trei criterii:

$$S(b) = \hat{c}(b) + \alpha_{area}(b) + \alpha_{temporal}(b) + \alpha_{source}(b)$$

1. **Confidența normalizată** $\hat{c}(b)$: scorul de detecție normalizat.
2. **Priorul de dimensiune** $\alpha_{area}$:
   - $-0.6$ dacă $\frac{w \times h}{W \times H} < 0.005$ (bbox prea mic)
   - $-0.5$ dacă $\frac{w \times h}{W \times H} > 0.70$ (bbox prea mare)
   - $+0.25$ altfel
3. **Consistența temporală** $\alpha_{temporal}$: dacă există un bbox selectat pe cadrul anterior:
$$\alpha_{temporal} = 0.8 \times IoU(b, b_{prev}) - 0.35 \times \frac{d_{center}(b, b_{prev})}{\max(W, H)}$$
   Acest termen favorizează bbox-urile care mențin o poziție și formă similară cu cadrul anterior.
4. **Bias de sursă** $\alpha_{source}$: $+0.10$ dacă sursa este HOG (preferință ușoară pentru detecțiile bazate pe formă față de cele bazate pe mișcare).

### 2.5. Netezirea Temporală (EMA Smoothing)

Bbox-ul câștigător pe fiecare cadru trece printr-o **netezire exponențială** (`smooth_bbox`) cu parametrul $\alpha = 0.65$:

$$x_t^{smooth} = \alpha \cdot x_{t-1}^{smooth} + (1 - \alpha) \cdot x_t^{raw}$$

Acest filtru EMA (Exponential Moving Average) elimină salturile bruște ale bbox-ului între cadre consecutive, producând o traiectorie fluidă a persoanei detectate.

Suplimentar, un mecanism de **temporal carry** menține ultimul bbox valid timp de până la **2 cadre consecutive** fără detecție (`miss_streak < 2`), prevenind golurile scurte din secvența de detecții.

### 2.6. Selecția Grupurilor Optime

După scanarea completă a videoclipului, funcția `select_centered_groups` selectează cele mai bune **10 grupuri temporale** din toate candidații posibili:

1. **Enumerarea candidaților**: Pentru fiecare cadru $c$ din video, se verifică dacă un grup centrat pe $c$ (cu offseturile $[-4, -2, 0, +2, +4]$) se încadrează în limitele videoclipului și dacă **toate cele 5 cadre** au detecție validă (bbox prezent cu calitate acceptabilă).

2. **Ordonarea după confidență**: Grupurile candidate sunt sortate descrescător după **scorul de confidență** al detecției pe cadrul central.

3. **Selecția non-overlapping (fără suprapunere)**: Grupurile sunt selectate greedy — după ce un grup este selectat, toate cadrele din zona sa temporală ($[c - 4, c + 4]$) sunt blocate, prevenind suprapunerea.

4. **Completare cu suprapunere**: Dacă nu se ating 10 grupuri non-overlapping, se permit grupuri cu centre noi (nefolosite), chiar dacă se suprapun parțial cu grupuri deja selectate.

5. **Sortare cronologică**: Grupurile selectate sunt sortate crescător după indexul cadrului central.

### 2.7. Statistici: 594 din 599 Videouri Procesate

Scriptul `extract_bboxes_kth.py` procesează cu succes **594 din cele 599 videoclipuri** din dataset-ul KTH, atingând o rată de acoperire de **99.2%**. Cele 5 videoclipuri rămase nu produc grupuri temporale valide (toate cele 5 cadre cu detecție) din cauza unor secvențe video excepțional de dificile (persoana aflată la distanță foarte mare de cameră în scenariul s2, sau mișcări foarte rapide cu blur extins).

Mecanismul dual HOG + MOG2 cu fuziune per-cadru este responsabil pentru această rată ridicată de acoperire. Într-o configurație doar cu HOG (fără MOG2), rata scade semnificativ sub 90%, multe videoclipuri de boxing și running rămânând fără detecții valide.

**[* FIGURA 2: Un tabel sau bar chart arătând distribuția detecțiilor per acțiune: câte videouri din fiecare acțiune sunt procesate cu succes, câte folosesc HOG primar și câte folosesc MOG2 ca contribuitor semnificativ. *]**

### 2.8. Output-ul JSON și Imaginile de Preview

Scriptul generează fișierul `hog_person_data_5.json` cu structura:
```json
{
  "config": {
    "temporal_kernel": 5,
    "num_groups": 10,
    "frame_gap": 2,
    "frame_width": 80,
    "frame_height": 60
  },
  "videos": {
    "train/boxing/person01_boxing_d1_uncomp.avi": {
      "total_video_frames": 447,
      "detection_frames": 312,
      "num_groups": 10,
      "detector": "hog",
      "groups": [
        [
          {"frame_idx": 42, "bboxes": [{"x": 15, "y": 8, "w": 30, "h": 45, "confidence": 0.82, "source": "hog+mog2"}], "selected_bbox": {...}, "is_center": false},
          {"frame_idx": 44, "bboxes": [...], "selected_bbox": {...}, "is_center": false},
          {"frame_idx": 46, "bboxes": [...], "selected_bbox": {...}, "is_center": true},
          {"frame_idx": 48, "bboxes": [...], "selected_bbox": {...}, "is_center": false},
          {"frame_idx": 50, "bboxes": [...], "selected_bbox": {...}, "is_center": false}
        ],
        ...
      ]
    }
  }
}
```

Fiecare bbox conține câmpul `"source"` indicând proveniența detecției: `"hog"`, `"mog2"`, sau `"hog+mog2"` (fuziune). De asemenea, scriptul salvează **imagini preview** în directorul `hog_previews_5/`, cu strip-uri orizontale de 5 cadre, bounding box-urile fiind desenate în **portocaliu** (`(255, 80, 0)`) pentru HOG și **verde** (`(0, 200, 0)`) pentru MOG2.

**[* FIGURA 3: 2–3 imagini din `hog_previews_5/` — una pentru boxing (posibil cu sursa hog+mog2), una pentru walking (cu HOG bbox portocaliu), una pentru running/jogging, arătând cele 5 cadre ale grupului cu bounding box-uri. Menționează sub imagine sursa detecției. *]**

### 2.9. Comanda de Rulare

```bash
python3 src/tool/extract_bboxes_kth.py \
    --temporal_kernel 5 \
    --num_groups 10 \
    --frame_gap 2 \
    --frame_width 80 \
    --frame_height 60 \
    --hit_threshold -0.75 \
    --mog2_fallback \
    --mog2_min_area 180 \
    --min_bbox_area_ratio 0.008 \
    --min_bbox_aspect 0.22 \
    --max_bbox_aspect 1.6
```

---

## 3. Clasele C++ pentru Pipeline-ul HOG-Guided: `VideoKTH_3D`

Clasa `VideoKTH_3D` (`include/dataset/VideoKTH_3D.h`, `src/dataset/VideoKTH_3D.cpp`) este componenta C++ care **citește cadrele video exact la indicii specificați în fișierul JSON** produs de `extract_bboxes_kth.py`, în loc să le extragă secvențial cu frame gap fix sau aleatoriu.

### 3.1. Parsarea JSON-ului HOG (`load_hog_json`)

La prima instanțiere a clasei (fie pentru train, fie pentru test), metoda `load_hog_json` citește integral fișierul `hog_person_data_5.json` folosind un **parser JSON manual** scris de la zero (fără dependențe externe precum nlohmann/json sau RapidJSON). Datele parsate sunt stocate într-o structură **statică partajată** între toate instanțele clasei:

```cpp
static std::map<std::string, VideoHOGData> _hog_data;
static bool _hog_data_loaded;  // flag one-shot — JSON-ul se citește o singură dată
```

Parserul procesează fișierul JSON caracter cu caracter, navigând prin structura ierarhică: secțiunea `"videos"` → fiecare cheie video (`"train/boxing/person01.avi"`) → `"total_video_frames"` + `"groups"` → fiecare grup (array de frame objects) → fiecare frame object cu `"frame_idx"` și `"bboxes"`.

### 3.2. Structuri de Date Interne

```cpp
struct FrameBBox {
    int frame_idx;                                       // indexul absolut al cadrului în video
    std::vector<std::tuple<int, int, int, int>> bboxes;  // (x, y, w, h) per detecție
};

struct VideoGroup {
    std::vector<FrameBBox> frames;  // 5 cadre per grup (temporal_kernel = 5)
};

struct VideoHOGData {
    int total_video_frames;         // nr. total de cadre din video
    std::vector<VideoGroup> groups; // până la 10 grupuri per video
};
```

**Exemplu concret** pentru `person01_boxing_d1_uncomp.avi`:
```
VideoHOGData {
    total_video_frames: 447,
    groups: [
        VideoGroup {  // Grupul 0 (primul, cel mai bun scor de confidență)
            frames: [
                FrameBBox { frame_idx: 42, bboxes: [(15, 8, 30, 45)] },   // F_c-4
                FrameBBox { frame_idx: 44, bboxes: [(16, 9, 29, 44)] },   // F_c-2
                FrameBBox { frame_idx: 46, bboxes: [(17, 9, 30, 44)] },   // F_center
                FrameBBox { frame_idx: 48, bboxes: [(18, 10, 29, 43)] },  // F_c+2
                FrameBBox { frame_idx: 50, bboxes: [(19, 10, 30, 43)] }   // F_c+4
            ]
        },
        VideoGroup { ... },  // Grupul 1
        ...                  // până la 10 grupuri
    ]
}
```

### 3.3. Sistemul de Mapare: Sample Index → (Video Key, Group Index)

Aceasta este **piesa centrală** care leagă clasa de dataset de clasa de sampling. Problema fundamentală: când `Convolution3D` apelează `HOGSampler3D::sample(... current_index ...)`, sampler-ul trebuie să știe **din ce video provine sample-ul curent** și **din ce grup temporal** (pentru a accesa bounding box-urile corecte). Dar sampler-ul nu are acces direct la informațiile de navigare ale dataset-ului.

Soluția: `VideoKTH_3D` construiește **două mapări statice** pe măsură ce generează sample-uri:

```cpp
static std::map<size_t, std::pair<std::string, size_t>> _train_sample_mapping;
static std::map<size_t, std::pair<std::string, size_t>> _test_sample_mapping;
static size_t _train_sample_counter;   // contor global train
static size_t _test_sample_counter;    // contor global test
```

La fiecare apel `next()`, imediat după identificarea videoclipului și a grupului, se înregistrează:

```cpp
if (_is_train)
    _train_sample_mapping[_train_sample_counter++] = {rel_key, group_idx};
else
    _test_sample_mapping[_test_sample_counter++] = {rel_key, group_idx};
```

**Exemplu de mapare generată** (cu `sample_per_video = 4`):

| Sample Index | Video Key | Group Index |
|-------------|-----------|-------------|
| 0 | `train/boxing/person01_boxing_d1_uncomp.avi` | 0 |
| 1 | `train/boxing/person01_boxing_d1_uncomp.avi` | 1 |
| 2 | `train/boxing/person01_boxing_d1_uncomp.avi` | 2 |
| 3 | `train/boxing/person01_boxing_d1_uncomp.avi` | 3 |
| 4 | `train/boxing/person01_boxing_d2_uncomp.avi` | 0 |
| 5 | `train/boxing/person01_boxing_d2_uncomp.avi` | 1 |
| ... | ... | ... |

Astfel, când `HOGSampler3D` primește `current_index = 5`, poate determina: „acest sample provine din `train/boxing/person01_boxing_d2_uncomp.avi`, grupul temporal 1" → consultă `_hog_data["train/boxing/person01_boxing_d2_uncomp.avi"].groups[1]` → obține cele 5 `FrameBBox`-uri cu coordonatele persoanei.

**Resetarea mapărilor**: Funcția `reset_sample_mappings()` este apelată explicit în `KTH_3D.cpp` **înainte** de crearea instanțelor de train/test, curățând contoarele și mapările de la eventuale rulări anterioare:

```cpp
dataset::VideoKTH_3D::reset_sample_mappings();
experiment.add_train<dataset::VideoKTH_3D>(...);
experiment.add_test<dataset::VideoKTH_3D>(...);
```

### 3.4. Extragerea Cadrelor: Metoda `next()`

Metoda `next()` este motorul principal al clasei. La fiecare apel, produce un **tensor de output** $[H \times W \times C \times T]$ conținând $T$ cadre video dintr-un grup temporal specific:

**Pasul 1 — Identificarea videoclipului:**
```cpp
_current_video_name = _video_list[_cursor];
```
Cursorul `_cursor` iterează prin lista sortată de videoclipuri. `_cursor_count` indică al câtelea sample din videoclipul curent (0 la `_sample_per_video - 1`).

**Pasul 2 — Conversia la cheie relativă:**
```cpp
std::string rel_key = get_relative_video_key(_current_video_name);
```
Transformă calea absolută (ex. `/home/mihai/kth_organized/train/boxing/person01.avi`) în cheia relativă din JSON (`train/boxing/person01.avi`), căutând prefixul `train/` sau `test/` în path.

**Pasul 3 — Lookup-ul grupului HOG:**
```cpp
const VideoHOGData &vdata = _hog_data[rel_key];
size_t group_idx = _cursor_count;  // al câtelea sample din video = al câtelea grup
const VideoGroup &group = vdata.groups[group_idx];
```
Se accesează grupul temporal cu indexul `_cursor_count` (primul sample → primul grup, al doilea sample → al doilea grup, etc.).

**Pasul 4 — Extragerea indicilor de cadre:**
```cpp
for (const auto &fb : group.frames)
    target_frame_indices.push_back(fb.frame_idx);
```
Se obțin indicii exacti ai cadrelor (ex. `[42, 44, 46, 48, 50]`).

**Pasul 5 — Citirea cadrelor din video prin seek:**
```cpp
capture.set(cv::CAP_PROP_POS_FRAMES, target);  // salt direct la cadrul dorit
capture >> frame;
cv::resize(frame, frame, cv::Size(80, 60));
cv::cvtColor(frame, frame, cv::COLOR_BGR2GRAY);
```
Spre deosebire de citirea secvențială (care ar necesita parcurgerea tuturor cadrelor intermediare), `CAP_PROP_POS_FRAMES` face un **seek direct** la indexul specificat, economisind timp de I/O.

**Pasul 6 — Stocarea în tensor:**
Fiecare cadru este stocat în dimensiunea temporală corespunzătoare a tensorului de output:
```
out.second.at(row, col, channel, frame_number) = pixel_value;
```
Rezultă un tensor $[60 \times 80 \times 1 \times 5]$ (H × W × C × T) pentru 5 cadre grayscale la rezoluția 80×60.

**Pasul 7 — Avansarea cursorului:**
```cpp
_cursor_count++;
if (_cursor_count >= _sample_per_video) {
    _cursor++;          // trece la următorul video
    _cursor_count = 0;  // resetează contorul de sample-uri
}
```

### 3.5. Fallback la Eșantionare Standard

Dacă fișierul JSON lipsește sau un videoclip nu are date HOG (nu se găsește `rel_key` în `_hog_data`), clasa revine automat la metoda standard de eșantionare cu frame gap (identică cu `dataset::Video`):
- Se sare `_cursor_count * 3` cadre de la începutul videoclipului
- Se citesc cadre consecutive cu frame gap fix
- Se aplică pragul de mișcare (`movement_threshold`) pentru a sări cadrele statice

Acest fallback garantează că experimentul poate rula chiar și fără fișierul JSON, dar cu performanțe inferioare.

**[* FIGURA 4: O diagramă flux arătând: `extract_bboxes_kth.py` → `hog_person_data_5.json` → `VideoKTH_3D::load_hog_json()` → `next()` face seek la cadrele specificate → Tensor $[60 \times 80 \times 1 \times 5]$. *]**

---

## 4. Arhitectura Sampler — Separarea Strategiei de Eșantionare (Strategy Pattern)

O contribuție arhitecturală importantă este **refactorizarea modului în care se extrag patch-urile** din sample-uri în timpul antrenării STDP. În versiunea originală a simulatorului, logica de sampling era **hardcoded** direct în clasa de convoluție — dacă se dorea o strategie diferită de eșantionare, trebuia creată o clasă de convoluție complet nouă, duplicând tot codul de antrenare, testare și vizualizare.

Refactorizarea aplică **Strategy Pattern**: logica de sampling este extrasă într-o ierarhie de clase `Sampler` independente, injectate în `Convolution3D` ca parametru configurabil.

**Ierarhia de clase:**
```
Sampler (clasă abstractă, include/Sampler.h)
├── RandomSampler3D  — sampling uniform aleatoriu (include/sampler/RandomSampler3D.h)
└── HOGSampler3D     — sampling ghidat de bounding box-uri HOG (include/sampler/HOGSampler3D.h)
```

**Clasa abstractă `Sampler`** (`include/Sampler.h`):
- Urmează exact pattern-ul `STDP` / `STDPFactory` din simulator — este o subclasă `ClassParameter` cu propria fabrică (`SamplerFactory`), integrată în sistemul de parametri al simulatorului.
- Definește metoda virtuală pură `sample()` care primește tensorul input, dimensiunile layerului, un generator random și indexul sample-ului curent, returnând un `Patch3D(x, y, k)` — coordonatele de unde se extrage patch-ul.
- `Patch3D` codifică: `x` = offset pe rânduri (vertical), `y` = offset pe coloane (orizontal), `k` = offset temporal.

### 4.1. `HOGSampler3D` — Eșantionare Ghidată de Bounding Box-uri (Detaliu Extrem)

`HOGSampler3D` (`include/sampler/HOGSampler3D.h`, `src/sampler/HOGSampler3D.cpp`) este sampler-ul care **focalizează eșantionarea STDP pe zona persoanei detectate**, fără nicio dependență de OpenCV la runtime. Acesta accesează datele statice pre-calculate din `VideoKTH_3D::get_hog_data()` și `VideoKTH_3D::get_train_sample_mapping()` / `get_test_sample_mapping()` pentru a plasa patch-urile de antrenare **exclusiv în interiorul bounding box-ului** persoanei.

#### 4.1.1. Structura `BoundingBox` și Cache-ul

HOGSampler3D menține un **cache intern** pentru a evita recalcularea coordonatelor de sampling la fiecare epocă:

```cpp
struct BoundingBox {
    bool has_person;    // persoana a fost detectată pe acest cadru
    size_t min_x;       // limita superioară (rând minim) pentru plasarea patch-ului
    size_t max_x;       // limita inferioară (rând maxim)
    size_t min_y;       // limita stângă (coloană minimă)
    size_t max_y;       // limita dreaptă (coloană maximă)
};

std::map<size_t, std::vector<BoundingBox>> _person_box_cache;
```

Cheia cache-ului este `current_index` (indexul global al sample-ului), iar valoarea este un **vector de BoundingBox-uri** — câte unul per cadru temporal din grup (5 elemente pentru `temporal_kernel = 5`).

**De ce un cache?** Convoluția STDP rulează **100 de epoci** (`epoch = 100`). La fiecare epocă, fiecare sample este procesat din nou, iar sampler-ul este apelat pentru fiecare sample. Fără cache, la fiecare apel s-ar repeta operațiile de lookup (căutare în mapare, în hog_data, conversie coordonate) — adică 100 × N_samples operații de lookup. Cu cache-ul, lookup-ul se face **o singură dată** per sample (la prima epocă), iar de la a doua epocă încolo se accesează direct cache-ul O(1).

#### 4.1.2. `ensure_cache_built()` — Inițializarea Lazy

La primul apel `sample()`, metoda `ensure_cache_built()` verifică dacă datele HOG sunt disponibile:

```cpp
void HOGSampler3D::ensure_cache_built() {
    if (_cache_built) return;
    const auto &hog_data = dataset::VideoKTH_3D::get_hog_data();
    // Raportează dacă datele sunt disponibile sau se va folosi fallback random
    _cache_built = true;
}
```

Aceasta nu populează cache-ul complet — popularea se face **lazy**, la primul acces per-sample, în `sample_point_inside_person()`.

#### 4.1.3. `sample()` — Fluxul Complet de Eșantionare

Semnătura metodei:
```cpp
Patch3D sample(const Tensor<float> &sample,
               size_t width, size_t height, size_t conv_depth,
               size_t filter_width, size_t filter_height, size_t filter_conv_depth,
               size_t current_index, std::default_random_engine &rng);
```

Unde:
- `width`, `height` = dimensiunile spațiale ale input-ului (ex. 80, 60 după OnOff → 80 pe lățime, 60 pe înălțime)
- `conv_depth` = adâncimea temporală a input-ului (ex. 5 cadre)
- `filter_width`, `filter_height`, `filter_conv_depth` = dimensiunile filtrului (ex. 5×5×2)
- `current_index` = indexul global al sample-ului curent (cheia de acces la maparea statică)

**Pasul 1 — Selectarea indexului temporal $k$:**
```cpp
if (filter_conv_depth < conv_depth)
    k = uniform_random(0, conv_depth - filter_conv_depth);
```
Cu `conv_depth = 5` (cadre) și `filter_conv_depth = 2` (adâncimea temporală a filtrului), $k$ este ales uniform aleatoriu din $\{0, 1, 2, 3\}$, determinând de la ce cadru temporal începe extragerea patch-ului. De exemplu, $k = 1$ înseamnă că filtrul procesează cadrele $[F_1, F_2]$.

**Pasul 2 — Verificarea adâncimii input-ului:**
```cpp
size_t input_depth = sample.shape().dim(2);  // nr. canale (ex. 2 pentru ON/OFF)
if (input_depth <= 3) {
    // Folosește eșantionare ghidată HOG
    auto [sample_x, sample_y] = sample_point_inside_person(
        width, height, filter_width, filter_height, rng, current_index, k);
} else {
    // Fallback la random (feature maps profunde)
    x = uniform_random(0, width - filter_width);
    y = uniform_random(0, height - filter_height);
}
```

**Condiția `input_depth <= 3`** este esențială: eșantionarea ghidată HOG se aplică **doar pe primul strat convoluțional** (conv1), unde input-ul are 1–3 canale (1 grayscale, 2 ON/OFF, sau 3 RGB). La straturile ulterioare (conv2, fc1), input-ul are zeci de canale (ex. 64 canale de feature maps de la conv1), iar:
1. Coordonatele bounding box-urilor au fost calculate pe imaginea originală — nu mai corespund spațiului de feature maps
2. Informația a fost deja focalizată pe zona persoanei de conv1
3. Eșantionarea random devine adecvată deoarece conv1 a „filtrat" fundalul irelevant

**Pasul 3 — Returnarea patch-ului:**
```cpp
return Patch3D(x, y, k);
```
Coordonatele $(x, y, k)$ indică pozițiile de start din tensorul de input de unde `Convolution3D` va extrage patch-ul de dimensiune $[\text{filter\_height} \times \text{filter\_width} \times \text{input\_depth} \times \text{filter\_conv\_depth}]$.

#### 4.1.4. `sample_point_inside_person()` — Mecanismul Intern Pas cu Pas

Aceasta este **funcția centrală** care transformă bounding box-urile din JSON în coordonate de sampling valide. Fluxul complet:

**Cazul 1 — Cache Hit (epocile 2+):**

Dacă `current_index` există deja în `_person_box_cache`:

```cpp
auto cache_it = _person_box_cache.find(current_index);
if (cache_it != _person_box_cache.end()) {
    const std::vector<BoundingBox> &boxes = cache_it->second;
    // Folosește direct cache-ul
}
```

Se verifică dacă cadrul temporal $k$ are persoană detectată:
- **Dacă DA**: $x \sim \mathcal{U}(\text{min\_x}^{(k)}, \text{max\_x}^{(k)})$, $y \sim \mathcal{U}(\text{min\_y}^{(k)}, \text{max\_y}^{(k)})$
- **Dacă NU**: se aplică **fallback-ul temporal** (detaliat mai jos în §4.1.6)

**Cazul 2 — Cache Miss (prima epocă):**

Dacă `current_index` NU există în cache, se realizează **lookup-ul complet**:

1. **Identificarea video-ului**: Se caută `current_index` mai întâi în `_train_sample_mapping`, apoi în `_test_sample_mapping`:
```cpp
auto it_tr = train_map.find(current_index);
if (it_tr != train_map.end()) {
    video_key = it_tr->second.first;   // ex: "train/boxing/person01.avi"
    group_idx = it_tr->second.second;  // ex: 2 (al treilea grup)
    found = true;
}
```

2. **Accesarea datelor HOG**: Se caută `video_key` în `_hog_data` și se obține grupul temporal cu indexul `group_idx`:
```cpp
const auto &group = hog_it->second.groups[group_idx];
```

3. **Conversia coordonatelor** (detaliată în §4.1.5).

4. **Popularea cache-ului**:
```cpp
_person_box_cache[current_index] = boxes;
```

5. **Returnarea coordonatelor** (identic cu Cazul 1).

#### 4.1.5. Conversia Coordonatelor Bbox → Interval de Sampling

Bounding box-ul din JSON este exprimat în **coordonate de imagine**: `{x, y, w, h}` unde `(x, y)` este colțul stânga-sus, `w` = lățime (pe orizontală), `h` = înălțime (pe verticală).

Patch-ul `Patch3D(x, y, k)` folosește o convenție diferită: `x` = offset pe **rânduri** (vertical, dim 0 a tensorului), `y` = offset pe **coloane** (orizontal, dim 1 a tensorului). Conversia:

```cpp
auto [bx, by, bw, bh] = fb.bboxes[0];  // bx = coloana, by = rândul

// Intervalul valid pentru patch.x (rânduri, vertical):
size_t min_x = max(0, by);                          // rândul de start al bbox
size_t max_x = min(W - fw, by + bh - fw);           // rândul maxim unde filtrul încape

// Intervalul valid pentru patch.y (coloane, orizontal):
size_t min_y = max(0, bx);                          // coloana de start a bbox
size_t max_y = min(H - fh, bx + bw - fh);           // coloana maximă unde filtrul încape
```

**Condiția de validitate**: `max_x >= min_x && max_y >= min_y`. Dacă bounding box-ul este **prea mic** pentru a conține filtrul (ex. bbox de 3×8 dar filtru de 5×5), bounding box-ul este marcat ca `has_person = false` — filtrul nu încape fizic în zona detectată.

**Exemplu numeric**: Pentru un bbox `{x=15, y=8, w=30, h=45}` cu filtru `5×5` pe un cadru `80×60`:
- `min_x = max(0, 8) = 8` (rândul 8)
- `max_x = min(80-5, 8+45-5) = min(75, 48) = 48` (rândul 48)
- `min_y = max(0, 15) = 15` (coloana 15)
- `max_y = min(60-5, 15+30-5) = min(55, 40) = 40` (coloana 40)

Sampling-ul va genera: $x \sim \mathcal{U}(8, 48)$, $y \sim \mathcal{U}(15, 40)$ — patch-ul de $5 \times 5$ va fi întotdeauna **complet conținut** în bounding box-ul persoanei.

**[* FIGURA 5: Un cadru KTH (80×60) cu bounding box-ul HOG desenat (dreptunghi portocaliu), și în interiorul acestuia mai multe dreptunghiuri mici de 5×5 pixeli (patch-uri posibile), arătând vizual zona din care HOGSampler3D extrage eșantioane. Compară cu un cadru unde patch-urile sunt dispersate pe tot cadrul (RandomSampler3D). *]**

#### 4.1.6. Fallback Temporal și Fallback Total

**Fallback Temporal** — Când cadrul $k$ nu are persoană detectată (fie bbox absent, fie bbox prea mic), sampler-ul caută **cel mai apropiat cadru vecin** cu detecție validă:

```cpp
int best_frame = -1;
int min_dist = boxes.size();
for (size_t f = 0; f < boxes.size(); f++) {
    if (!boxes[f].has_person) continue;
    int dist = abs((int)f - (int)temporal_index);
    if (dist < min_dist) {
        min_dist = dist;
        best_frame = (int)f;
    }
}
```

Acest mecanism garantează că, chiar dacă HOG/MOG2 nu a detectat persoana pe un cadru specific din grup, patch-ul STDP va fi extras totuși din zona persoanei, folosind coordonatele unui cadru vecin. Într-o secvență video la 25 fps cu frame_gap = 2, cadrele vecine sunt suficient de apropiate temporal încât persoana să fie aproximativ în aceeași zonă.

**Fallback Total** — Dacă **niciun cadru** din grup nu are detecție validă (extrem de rar, apare la <1% din sample-uri):
```cpp
// Dacă nu s-a găsit nicio detecție: random uniform pe tot cadrul
std::uniform_int_distribution<size_t> fallback_x(0, W - fw);
std::uniform_int_distribution<size_t> fallback_y(0, H - fh);
return {fallback_x(rng), fallback_y(rng)};
```

**Ierarhia completă de fallback:**
1. ✅ Bbox valid pe cadrul temporal $k$ → sampling în interiorul bbox-ului
2. ⚠️ Bbox absent pe $k$, dar prezent pe un cadru vecin → sampling folosind bbox-ul celui mai apropiat cadru
3. ❌ Niciun bbox valid în niciun cadru → random uniform pe tot cadrul (echivalent cu `RandomSampler3D`)

**[* FIGURA 6: O diagramă cu cele 3 niveluri de fallback, arătând pentru fiecare: condiția de activare, sursa coordonatelor, și frecvența estimată de apariție. *]**

### 4.2. `RandomSampler3D` — Eșantionare Aleatorie

`RandomSampler3D` (`include/sampler/RandomSampler3D.h`, `src/sampler/RandomSampler3D.cpp`) este sampler-ul implicit, care selectează coordonatele `(x, y, k)` **uniform aleatoriu** din spațiul disponibil, **fără nicio ghidare de la bounding box-uri HOG**:

```cpp
Patch3D RandomSampler3D::sample(...) {
    size_t x = 0, y = 0, k = 0;
    if (filter_width < width)
        x = uniform_random(0, width - filter_width);
    if (filter_height < height)
        y = uniform_random(0, height - filter_height);
    if (filter_conv_depth < conv_depth)
        k = uniform_random(0, conv_depth - filter_conv_depth);
    return Patch3D(x, y, k);
}
```

Fiecare dimensiune este eșantionată **independent**:
- $x \sim \mathcal{U}(0, W - f_w)$ — orice rând valid
- $y \sim \mathcal{U}(0, H - f_h)$ — orice coloană validă
- $k \sim \mathcal{U}(0, T - f_t)$ — orice moment temporal valid

**Când se folosește RandomSampler3D:**
1. **Ca alternativă la HOGSampler3D pe conv1** — pentru experimente de comparare (baseline):
```cpp
conv1.parameter<Sampler>("sampler").set<sampler::RandomSampler3D>();
```
2. **Pe straturile ulterioare** (conv2, fc1) — unde input-ul este deja focalizat pe persoană de conv1 și ghidarea HOG nu mai este necesară/posibilă.

**Dezavantajul pe conv1**: Fără ghidare HOG, STDP-ul va extrage ~50%+ din patch-uri din regiunile de fundal (cer, pardoseală, perete), ducând la filtre care învață textura fundalului în loc de anatomia persoanei. Acest efect este numit „orbirea filtrelor" și este principala motivație pentru introducerea HOGSampler3D.

**[* FIGURA 7: Comparație vizuală a filtrelor (weights) învățate cu HOGSampler3D vs. RandomSampler3D pe conv1 — arătând cum filtrele HOG-guided capturează contururi anatomice (brațe, picioare, trunchi) în timp ce filtrele random capturează și texturi de fundal. *]**

### 4.3. `ConvolutionSampler3D` — Wrapper Semantic pentru Experimente

`ConvolutionSampler3D` (`include/layer/ConvolutionSampler3D.h`, `src/layer/ConvolutionSampler3D.cpp`) este un **wrapper subțire** (thin wrapper) peste `Convolution3D`, introdus pentru a păstra un nume semantic distinct în contextul experimentelor sampler-driven:

```cpp
class ConvolutionSampler3D : public Convolution3D {
public:
    ConvolutionSampler3D();
    ConvolutionSampler3D(size_t filter_number, size_t filter_width,
                         size_t filter_height, size_t filter_depth,
                         std::string model_path = "",
                         size_t stride_x = 1, size_t stride_y = 1, size_t stride_k = 1,
                         size_t padding_x = 0, size_t padding_y = 0, size_t padding_k = 0);
};
```

**Comportamentul este identic cu `Convolution3D`** — moștenește toate funcționalitățile (antrenare STDP, testare, WTA, vizualizare ponderi) fără nicio logică proprie. Constructorii delegă direct către `Convolution3D`.

**Diferența notabilă**: Ordinea parametrilor constructorului este diferită:
- `Convolution3D(filter_width, filter_height, filter_depth, filter_number, ...)`
- `ConvolutionSampler3D(filter_number, filter_width, filter_height, filter_depth, ...)`

**Utilizare în experiment:**
```cpp
// Varianta 1: Convolution3D direct (folosită curent)
auto &conv1 = experiment.push<layer::Convolution3D>(5, 5, 2, 64, "", 1, 1, 1);

// Varianta 2: ConvolutionSampler3D (alternativă semantică)
auto &conv1 = experiment.push<layer::ConvolutionSampler3D>(64, 5, 5, 2, "", 1, 1, 1);
```

Ambele variante produc **exact același rezultat** — un layer convoluțional cu 64 de filtre de 5×5 spațial și adâncime temporală 2, cu stride 1×1×1. Diferența este doar de semantică și organizare a codului.

**Înregistrarea în fabrică**: `ConvolutionSampler3D` este înregistrat independent în `LayerFactory`, permițând crearea sa prin configurare:
```cpp
static RegisterClassParameter<ConvolutionSampler3D, LayerFactory> _register("ConvolutionSampler3D");
```

**Scopul arhitectural**: Separarea permite ca experimentele care testează diferite strategii de sampling (HOG vs. Random) să fie identificabile vizual în cod prin numele layer-ului, fără a modifica `Convolution3D` originală.

**[* FIGURA 8: O diagramă UML simplificată arătând relația de moștenire: `Layer4D` → `Convolution3D` → `ConvolutionSampler3D`, cu notă că ConvolutionSampler3D nu adaugă logică nouă, doar un nume semantic distinct. *]**

*(Notă: în versiunile ulterioare ale experimentului, straturile convoluționale succesive (conv2, conv3...) vor folosi strategii de sampling diferite, unde procesarea eșantioanelor se va face diferit față de conv1. Această parte va fi adăugată ulterior.)*

### 4.4. Configurarea Sampler-ului ca Parametru al Layer-ului

Sampler-ul este injectat în `Convolution3D` prin sistemul de parametri al simulatorului, identic cu STDP-ul sau orice alt sub-obiect:

```cpp
// Eșantionare ghidată HOG (focalizată pe persoană)
conv1.parameter<Sampler>("sampler").set<sampler::HOGSampler3D>();

// Eșantionare random (nedirecționată)
conv1.parameter<Sampler>("sampler").set<sampler::RandomSampler3D>();
```

**Avantajele acestei arhitecturi:**
1. **Extensibilitate**: Pentru o nouă strategie de sampling, se creează doar o clasă nouă care moștenește `Sampler`, fără a modifica `Convolution3D`.
2. **Eliminarea duplicării**: Fosta clasă `HOG_Convolution3D` duplica ~800 linii de cod din `Convolution3D` doar pentru a schimba ~20 linii de sampling. Acum există o singură clasă de convoluție.
3. **Configurabilitate**: Sampler-ul este un parametru al layer-ului, interschimbabil cu o singură linie de cod.
4. **Separarea responsabilităților**: `VideoKTH_3D` = **ce date** se încarcă (video → tensori), `Sampler` = **de unde** se extrag patch-uri din acele date, `Convolution3D` = **cum** se antrenează filtrele pe patch-urile extrase.

**[* FIGURA 9: O diagramă flux completă arătând fluxul de date: `extract_bboxes_kth.py` → `hog_person_data_5.json` → `VideoKTH_3D` (încarcă cadrele + maparea sample) → `Convolution3D::process_train_sample()` cere patch → `HOGSampler3D::sample()` consultă maparea + cache-ul → returnează `Patch3D(x, y, k)` din zona persoanei → `Convolution3D::train()` antrenează filtrul STDP pe patch-ul extras. *]**

---

## 5. Extragerea Contrastului Spațial și Codificarea Biologică

Rețeaua Spiking nu lucrează cu pixeli reali pe o scară cromatică, ci traducând informația vizuală într-o reprezentare biologică asemănătoare cu cea a retinei umane. Acest proces de preprocesare transformă cadrele video brute în **timpi de latență** (spike-uri) printr-un pipeline de trei pași: filtrare DoG → scalare → codificare temporală.

### 5.1. `DefaultOnOffFilter` — Emularea Câmpului Receptor al Retinei

**Motivația biologică**: În retina umană, celulele ganglionare au câmpuri receptive **center-surround** (centru-periferie) antagoniste [34]. O celulă ON-center răspunde la un punct luminos pe fundal întunecat (centru excitator, periferie inhibitoare), iar o celulă OFF-center răspunde la un punct întunecat pe fundal luminos (centru inhibitor, periferie excitatoare). Această structură extrage natural marginile și contrastul local, ignorând zonele uniform iluminate.

**Implementarea matematică**: Câmpul receptor center-surround este modelat prin filtrul *Difference of Gaussians (DoG)* [39], calculat analitic ca diferența a două distribuții gaussiene cu deviații standard diferite:

$$DoG(x, y) = G_{\sigma_c}(x, y) - G_{\sigma_s}(x, y) = \frac{1}{2\pi \sigma_c^2} e^{-\frac{x^2+y^2}{2\sigma_c^2}} - \frac{1}{2\pi \sigma_s^2} e^{-\frac{x^2+y^2}{2\sigma_s^2}}$$

Cu parametrii experimentali:
- **Fereastră**: $7 \times 7$ pixeli (`filter_size = 7`)
- **Deviație centru**: $\sigma_c = 1.0$ (răspuns spațial îngust, focalizat pe detalii fine)
- **Deviație periferie**: $\sigma_s = 4.0$ (răspuns spațial larg, capturează contextul)
- **Raportul** $\sigma_s / \sigma_c = 4.0$ (valoare tipică pentru celulele ganglionare ale retinei)

**Convoluția cu imaginea**: Filtrul DoG este aplicat prin convoluție 2D pe fiecare cadru grayscale individual:
$$R(x, y) = (I * DoG)(x, y) = \sum_{i=-3}^{3} \sum_{j=-3}^{3} I(x+i, y+j) \cdot DoG(i, j)$$

Rezultatul $R(x,y)$ poate fi **pozitiv** (tranziție întuneric→lumină) sau **negativ** (tranziție lumină→întuneric).

**Separarea în canale ON/OFF**: Ieșirea este descompusă în două canale complementare prin rectificare:
- **Canalul ON** (canal 0): Răspunde la tranzițiile de la întuneric la lumină (marginile luminate):
$$I_{ON}(x, y) = \max(0, R(x, y))$$
- **Canalul OFF** (canal 1): Răspunde la tranzițiile de la lumină la întuneric (marginile umbrite):
$$I_{OFF}(x, y) = \max(0, -R(x, y))$$

Prin această descompunere, fiecare margine din imagine generează activitate pe **exact un canal** — o margine cu o parte luminoasă și una întunecată activează canalul ON pe partea luminoasă și canalul OFF pe partea întunecată, recreând informația completă a conturului.

**Transformarea dimensională**: Fiecare cadru grayscale cu 1 canal este transformat în 2 canale (ON + OFF):

$$[60 \times 80 \times 1 \times 5] \xrightarrow{DoG} [60 \times 80 \times 2 \times 5]$$

**[* FIGURA 10: O comparație vizuală pe 3 coloane: 1. Cadrul original alb-negru (80×60). 2. Harta canalului ON (contururi albe pe fundal negru — marginile luminate). 3. Harta canalului OFF (contururi albe pe fundal negru — marginile umbrite). Observă cum cele două canale sunt complementare — sumând ON + OFF se reconstruiește aproximativ imaginea de contururi completă. *]**

### 5.2. `MaxScaling` — Normalizarea Contrastului

Valorile absolute ale filtrului DoG variază în funcție de contrastul local al imaginii. `MaxScaling` normalizează toate valorile la intervalul $[0, 1]$ prin împărțirea la valoarea maximă per sample:

$$I_{norm}(x, y, c, t) = \frac{I(x, y, c, t)}{\max_{x', y', c', t'} I(x', y', c', t') + \epsilon}$$

Această normalizare asigură că toate sample-urile au aceeași amplitudine maximă, indiferent de iluminarea scenei (un cadru filmat în interior cu lumină slabă va avea aceeași scară cu unul filmat în exterior cu soare puternic).

### 5.3. `LatencyCoding` — Conversia Contrastului în Timpi de Spike

Aceasta este etapa în care datele **părăsesc domeniul analog** și intră în **domeniul spiking**. Fiecare pixel cu o valoare de contrast $I \in [0, 1]$ este transformat într-un **timp de latență** $t_s$ — momentul la care acel pixel va emite un spike:

$$t_s(x, y, c) = \max(0,\; 1 - I(x, y, c))$$

**Interpretare**:
- Un pixel cu contrast **puternic** ($I = 1.0$): $t_s = 0$ — emite spike-ul **cel mai devreme** (prima dată când neuronul convoluțional integrează)
- Un pixel cu contrast **slab** ($I = 0.2$): $t_s = 0.8$ — emite spike-ul **târziu**
- Un pixel cu contrast **zero** ($I = 0.0$): $t_s = 1.0$ — emite spike-ul **ultimul** (sau practic nu contribuie la integrare)

Această codificare implementează principiul biologic al **codificării prin rang** (rank-order coding) [9]: marginile cele mai pronunțate (cu cel mai mare contrast DoG) semnalizează primele, iar spike-urile lor sunt cele care influențează cel mai mult neuronii convoluționali, deoarece ajung primele la integrare.

**Input/Output**: $[60 \times 80 \times 2 \times 5]$. Valorile sunt acum timpi de latență, nu intensități — tensorul este pregătit pentru procesarea spiking în `Convolution3D`.

---

## 6. Stratul Convoluțional Spiking: Convolution3D — Antrenare și Testare

### 6.1. Memoria Spatio-Temporală

Setarea `filter_conv_depth = 2` în conv1 (parametrul `tmp_filter_size` din `KTH_3D.cpp`) forțează design-ul convoluției să asimileze **simultan câte 2 cadre temporale** din cele 5 disponibile. Filtrul acționează pe un volum $V = (x, y, t)$ de $5 \times 5 \times 2$, preluând asincron **2 cadre adiacente** dintr-o grupă la pachet.

Cu 5 cadre temporale la intrare și `filter_depth = 2`, dimensiunea temporală produce **4 output-uri temporale** la conv1:
$$T_{out} = \frac{5 - 2}{1} + 1 = 4$$

Rețeaua memorează dinamic animația („Optical Flow"), comparând conturul persoanei pe perechile de cadre temporale $\{(F_0, F_1), (F_1, F_2), (F_2, F_3), (F_3, F_4)\}$. Această configurare permite conv1 să detecteze **deplasarea conturului** și **viteza mișcării** direct din compararea perechilor de momente temporale.

### 6.2. Procesul de Antrenare: `process_train_sample()` + `train()`

Antrenarea CSNN-ului este un proces în **mai multe epoci**. La fiecare epocă, fiecare sample de antrenare este procesat printr-un ciclu complet de sampling → extragere patch → integrare spike-uri → actualizare STDP.

#### 6.2.1. Ciclul de Antrenare per Epocă

```
Pentru fiecare epocă e = 0 ... 99:
    Pentru fiecare sample s din dataset-ul de antrenare:
        1. Sampler-ul selectează Patch3D(x, y, k)        ← HOGSampler3D / RandomSampler3D
        2. Se extrage patch-ul din tensorul sample-ului     ← sub-volum 5×5×2×2
        3. Se convertesc spike-urile din patch în ordine temporală
        4. Se procesează spike-urile prin integrare neuronală
        5. La depășirea pragului → STDP actualizează ponderile
    La finalul epocii:
        - Annealing: lr = lr × 0.95
        - Adaptare praguri
```

#### 6.2.2. Extragerea Patch-ului (Pasul 1-2)

`Convolution3D::process_train_sample()` apelează sampler-ul pentru a obține coordonatele patch-ului:

```cpp
Patch3D patch = _sampler->sample(sample, width, height, conv_depth,
                                 filter_width, filter_height, filter_conv_depth,
                                 current_index, rng);
```

Din tensorul input $[60 \times 80 \times 2 \times 5]$ se extrage un **sub-volum** (patch) de dimensiune $[f_h \times f_w \times C_{in} \times f_t] = [5 \times 5 \times 2 \times 2]$, începând de la poziția $(patch.x, patch.y, 0, patch.k)$:

$$\text{input\_time}(dy, dx, c, t) = \text{sample}(patch.x + dy,\; patch.y + dx,\; c,\; patch.k + t)$$

pentru $dy \in [0, f_h)$, $dx \in [0, f_w)$, $c \in [0, C_{in})$, $t \in [0, f_t)$.

Acest patch conține **100 valori de latență** ($5 \times 5 \times 2 \times 2 = 100$), fiecare reprezentând momentul temporal la care acel pixel/canal/cadru emite un spike.

#### 6.2.3. Integrarea Spike-urilor și Descărcarea Neuronală (Pasul 3-4)

Spike-urile din patch sunt **sortate temporal** (cele cu latența cea mai mică primele) și integrate secvențial în potențialul de membrană al fiecăruia dintre cele 64 de neuroni (filtre):

$$V_j = V_j + w_j(dx, dy, c, t) \quad \text{dacă spike-ul de la pozitia } (dx, dy, c, t) \text{ sosește la momentul curent}$$

La fiecare pas de integrare, se verifică dacă vreun neuron a atins pragul:

$$\text{dacă } V_j \geq \theta_j \implies \text{neuronul } j \text{ trage (fire)}$$

**Primul neuron care trage** (cel cu cea mai rapidă acumulare) devine **câștigătorul** competiției locale și declanșează:
1. **Actualizarea STDP** a ponderilor sale
2. **Inhibiția laterală** a celorlalți neuroni (WTA)
3. **Adaptarea pragurilor** tuturor neuronilor

#### 6.2.4. Regula STDP — Biological

Varianta `Biological` a STDP (folosită în experiment cu `w_lr = 0.1`) actualizează ponderile neuronului câștigător pe baza diferenței temporale dintre spike-ul de intrare și momentul descărcării:

$$\Delta w_{ij} = \begin{cases} A_+ \cdot e^{-|\Delta t| / \tau_+} & \text{dacă } \Delta t > 0 \quad \text{(LTP — spike-ul de intrare a precedat descărcarea)} \\ -A_- \cdot e^{-|\Delta t| / \tau_-} & \text{dacă } \Delta t \leq 0 \quad \text{(LTD — spike-ul de intrare a urmat descărcării)} \end{cases}$$

**Interpretare practică**: Pixelii cu contrast mare (latență mică $t_s \approx 0$) care au contribuit cauzal la descărcarea neuronului vor avea ponderile **întărite** — neuronul „memorează" acele margini. Pixelii cu contrast slab care au emis spike-uri tardive vor avea ponderile **slăbite** — neuronul „uită" zonele irelevante. Prin repetarea pe zeci de epoci, fiecare filtru devine specializat pe un tip specific de contur/mișcare.

**[* FIGURA 11: Graficul STDP clasic (curba cu zona pozitivă LTP sus și negativă LTD jos, în funcție de $\Delta t$). *]**

#### 6.2.5. Adaptarea Pragurilor (Homeostazia)

La fiecare descărcare, pragurile **tuturor** neuronilor sunt ajustate:

- **Neuronul câștigător** $j$: pragul **crește** proporțional cu rata de învățare:
$$\theta_j \leftarrow \theta_j + lr_{th} \cdot (1 - t_{obj})$$

- **Ceilalți neuroni** $j' \neq j$: pragul **scade** ușor:
$$\theta_{j'} \leftarrow \max(\theta_{j'} - lr_{th} \cdot t_{obj} / N_{filtre},\; min_{th})$$

unde $t_{obj} = 0.75$ este target-ul de activare, $lr_{th} = 1.0$ rata de învățare a pragurilor, și $min_{th} = 1.0$ limita inferioară. Acest mecanism de **homeostazie** asigură că:
- Neuronii care trag prea des își cresc pragul (devin mai selectivi)
- Neuronii care nu trag niciodată își scad pragul (devin mai sensibili)
- Pe termen lung, toți neuronii ating o rată de activare apropiată de $t_{obj}$

#### 6.2.6. Inhibiția WTA (Winner-Takes-All)

Când un neuron trage pe o anumită poziție spațială, **toți ceilalți neuroni sunt inhibați** pe acea poziție — potențialele lor de membrană sunt resetate la 0. Acest mecanism forțează:
- **Diversitatea filtrelor**: Fiecare filtru trebuie să se specializeze pe un pattern diferit, deoarece nu poate câștiga competiția pe un pattern deja „revendicat" de alt filtru
- **Sparsity**: La orice moment, cel mult un neuron este activ per poziție spațială

În experimentul curent, `wta_infer = true` activează WTA și la testare (inferență), nu doar la antrenare.

#### 6.2.7. Annealing-ul Ratei de Învățare

La sfârșitul fiecărei epoci, rata de învățare globală scade exponențial:
$$lr^{(e+1)} = lr^{(e)} \times \alpha_{anneal}$$

Cu $\alpha_{anneal} = 0.95$, după 100 de epoci rata ajunge la:
$$lr^{(100)} = lr^{(0)} \times 0.95^{100} \approx lr^{(0)} \times 0.0059$$

Acest annealing permite rețelei să facă ajustări mari la început (explorare) și ajustări fine la sfârșit (convergență).

### 6.3. Procesul de Testare: `test()` + `forward()`

La testare, ponderile sinaptice sunt **înghețate** (nu se mai aplică STDP). Fiecare sample de test este propagat prin convoluție pentru a genera **hărți de trăsături** (feature maps):

1. **Propagarea spike-urilor** (`forward`): Tensorul de input complet $[60 \times 80 \times 2 \times 5]$ este parcurs prin fereastră glisantă la fiecare poziție spațială $(x, y)$ și temporală $(k)$. La fiecare poziție, se calculează potențialul fiecărui neuron:
$$V_j(x, y, k) = \sum_{t=0}^{f_t-1} \sum_{c=0}^{C-1} \sum_{dy=0}^{f_h-1} \sum_{dx=0}^{f_w-1} w_j(dx, dy, c, t) \cdot s(x+dx, y+dy, c, k+t)$$

2. **Descărcarea**: Dacă $V_j(x, y, k) \geq \theta_j$, neuronul $j$ emite un spike la poziția $(x, y, k)$ în tensorul de output $[56 \times 76 \times 64 \times 4]$.

3. **WTA la inferență**: Cu `wta_infer = true`, dacă mai mulți neuroni ating pragul la aceeași poziție spațială, doar neuronul cu potențialul cel mai mare trage. Aceasta produce feature maps **sparse** și **discriminative**.

4. **Contorizarea spike-urilor**: Se numără spike-urile totale emise de fiecare neuron pe întregul sample. Neuronii care nu emit niciodată (dead neurons) sunt raportați în statisticile de activitate.

**[* FIGURA 11b: O diagramă arătând diferența între antrenare (STDP activ, un singur patch per sample) și testare (STDP înghețat, convoluție completă pe tot tensorul). *]**

---

## 7. Orchestrarea Experimentului: Clasa `Experiment` și Infrastructura de Rulare

### 7.0a. Mediul de Rulare și Sistemul de Build

Experimentele sunt rulate pe o **mașină virtuală Google Cloud Platform (GCP)** de tip `c2-standard-8`:

| Componentă | Specificație |
|-----------|-------------|
| vCPU | 8 × Intel Cascade Lake (3.1 GHz) |
| RAM | 32 GB DDR4 |
| Disc | 100 GB SSD persistent |
| OS | Ubuntu 22.04 LTS |
| Compilator | GCC 11+ cu C++17 |
| Optimizări | `-march=native -msse -msse2 -msse3` (SIMD) |

**Sistemul de build** folosește **CMake** (versiunea minimă 3.10). Proiectul este organizat ca o bibliotecă partajată (`CSNNS`) linkată cu dependențele:

| Dependență | Utilizare |
|-----------|-----------|
| **OpenCV** | Citirea videoclipurilor (`VideoCapture`), redimensionare, conversie grayscale |
| **BLAS / LAPACK** | Operații de algebră liniară (pentru SVM și analize) |
| **libsvm** | Clasificatorul SVM (compilat din surse, `dep/libsvm/`) |
| **TBB** (Threading Building Blocks) | Paralelizare thread-safe a operațiilor |
| **pthreads** | Threading de bază |
| Qt4 *(opțional)* | Vizualizare GUI în timp real (dezactivat pe VM: `USE_GUI=OFF`) |

**Compilarea și rularea:**
```bash
# Pe mașina virtuală GCP:
mkdir csnn-simulator-build && cd csnn-simulator-build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j8 KTH_3D

# Rularea experimentului:
./KTH_3D
```

Executabilul `KTH_3D` este generat automat din `apps/kth/KTH_3D.cpp` prin mecanismul CMake care iterează toate fișierele din `apps/`:
```cmake
file(GLOB_RECURSE apps apps/*)
foreach(app ${apps})
    get_filename_component(app_name ${app} NAME_WE)
    add_executable(${app_name} ${app})
    target_link_libraries(${app_name} ${LIBRARY_NAME} ... )
endforeach()
```

### 7.0b. Clasa `Experiment` — Orchestratorul Pipeline-ului

Clasa template `Experiment<ExecutionEngine>` este **punctul de intrare** al simulatorului CSNN [27] care leagă toate componentele (dataset, preprocesare, layere convoluționale, analize) într-un pipeline executabil. Parametrul template specifică motorul de execuție — în cazul nostru `SparseIntermediateExecution`, optimizat pentru tensori sparse.

**Instanțierea experimentului** (`KTH_3D.cpp`):
```cpp
Experiment<SparseIntermediateExecution> experiment(
    argv, argc,
    "result/seed_123",     // directorul de rezultate
    "model/seed_123",      // directorul de modele salvate
    "kth_123",             // numele experimentului
    123,                   // seed random pentru reproductibilitate
    true,                  // verbose
    false, false           // flags de debug
);
```

**Construirea pipeline-ului** se face prin apeluri succesive care adaugă componente:

| Metodă | Ce face | Exemplu |
|--------|---------|---------|
| `experiment.push<Preprocessing>(...)` | Adaugă un strat de preprocesare | `push<DefaultOnOffFilter>(7, 1.0, 4.0)` |
| `experiment.add_train<Dataset>(...)` | Adaugă dataset-ul de antrenare | `add_train<VideoKTH_3D>(path, json, ...)` |
| `experiment.add_test<Dataset>(...)` | Adaugă dataset-ul de test | `add_test<VideoKTH_3D>(path, json, ...)` |
| `experiment.push<Layer>(...)` | Adaugă un strat convoluțional | `push<Convolution3D>(5, 5, 2, 64, ...)` |
| `experiment.output<Output>(layer, ...)` | Conectează o ieșire la un strat | `output<TimeObjectiveOutput>(conv1, 0.75)` |
| `out.add_postprocessing<P>(...)` | Adaugă postprocessing la ieșire | `add_postprocessing<SumPooling>(2, 2)` |
| `out.add_analysis<A>(...)` | Adaugă o analiză | `add_analysis<Svm>()` |
| `experiment.run(max_samples)` | **Execută** întregul experiment | `run(10000)` |

**`experiment.run(10000)`** declanșează secvența completă:
1. Încărcarea dataset-urilor (train + test) cu preprocesare aplicată automat
2. Antrenarea STDP pe fiecare strat convoluțional (100 epoci per strat)
3. Testarea (propagare forward cu ponderi înghețate) pe train + test
4. Postprocessing-ul feature maps-urilor (SumPooling, FeatureScaling)
5. Analizele finale (Activity, Coherence, SVM)
6. Salvarea ponderilor în format JSON în `Weights/kth_123/`

Parametrul `max_samples = 10000` limitează numărul maxim de sample-uri procesate (în practică, dataset-ul KTH are ~2400 sample-uri, deci limita nu este atinsă).

### 7.0c. Parametrii Experimentului Curent (`KTH_3D.cpp`)

Acest subcapitol fixează fluxul dimensional (tensorial) și parametrii exacti din experimentul curent.

### 7.1. Parametri Globali

| Parametru | Valoare | Semnificație |
|-----------|---------|-------------|
| `seed` | 123 | Seed-ul random pentru reproductibilitate |
| `frame_size_width` | 80 | Lățimea cadrului (pixeli) |
| `frame_size_height` | 60 | Înălțimea cadrului (pixeli) |
| `video_frames` | 5 | Numărul de cadre per grup temporal (temporal_kernel) |
| `frame_gap` | 0 | Frame gap la nivel de experiment (cadrele sunt deja selectate de JSON) |
| `grey` | 1 | Conversie grayscale activă |
| `threshold` | 5 | Prag de mișcare pentru fallback |
| `train_sample_per_video` | 4 | Câte sample-uri per video la antrenare |
| `test_sample_per_video` | 4 | Câte sample-uri per video la testare |

### 7.2. Preprocesare

| Strat | Parametri | Input → Output |
|-------|-----------|----------------|
| `DefaultOnOffFilter` | kernel=7, $\sigma_c$=1.0, $\sigma_s$=4.0 | $[60 \times 80 \times 1 \times 5]$ → $[60 \times 80 \times 2 \times 5]$ |
| `MaxScaling` | — | $[60 \times 80 \times 2 \times 5]$ → $[60 \times 80 \times 2 \times 5]$ |
| `LatencyCoding` | — | $[60 \times 80 \times 2 \times 5]$ → $[60 \times 80 \times 2 \times 5]$ |

### 7.3. Stratul `conv1` (Convolution3D + HOGSampler3D)

| Parametru | Valoare | Semnificație |
|-----------|---------|-------------|
| Filtru spațial | $5 \times 5$ | Fereastră de extracție spațială |
| Adâncime temporală | 2 (`tmp_filter_size`) | Procesează 2 cadre temporale simultan |
| Stride | $1 \times 1 \times 1$ | Pas unitar pe toate axele |
| Număr filtre | **64** | Numărul de pattern-uri învățate |
| Epoci | **100** | Numărul de treceri prin dataset |
| Annealing | $0.95$ | Factor de scădere a learning rate-ului |
| `min_th` | $1.0$ | Limita inferioară a pragului |
| `t_obj` | $0.75$ | Target-ul de activare per neuron |
| `lr_th` | $1.0$ | Rata de învățare a pragurilor |
| `wta_infer` | **true** | Winner-Takes-All activ la inferență |
| Inițializare ponderi | $\mathcal{U}(0, 1)$ | Distribuție uniformă |
| Inițializare threshold | $\mathcal{N}(9.0, 0.1)$ | Distribuție gaussiană |
| STDP | `Biological(0.1, 0.1)` | Regula de învățare cu $w_{lr} = 0.1$ |
| Sampler | **HOGSampler3D** | Eșantionare ghidată pe zona persoanei |

**Dimensiuni:**
- **Input**: Tensor Spike-uri din preprocesor: $[60 \times 80 \times 2 \times 5]$
- **Output**: $[56 \times 76 \times 64 \times 4]$
  - Spațial: $(60-5)/1+1 = 56$ rânduri, $(80-5)/1+1 = 76$ coloane
  - Canale: 64 filtre
  - Temporal: $(5-2)/1+1 = 4$ pași temporali
- **Sinapse per neuron**: $5 \times 5 \times 2 \times 2 = 100$

### 7.4. Output, Postprocessing și Clasificare

#### `TimeObjectiveOutput` — Extragerea Hărților de Trăsături

`TimeObjectiveOutput` este stratul care **colectează spike-urile de ieșire** ale conv1 pentru fiecare sample și le organizează într-un tensor de feature maps. Parametrul $t_{obj} = 0.75$ controlează **fereastra temporală de integrare** — spike-urile emise de neuroni în primele 75% din fereastra de integrare sunt înregistrate, cele ulterioare sunt ignorate. Acest prag asigură că doar spike-urile rapide (cele mai relevante temporal) contribuie la reprezentarea finală.

**Output per sample**: Un tensor $[56 \times 76 \times 64 \times 4]$ unde fiecare element este **1** (spike emis de filtrul $j$ la poziția $(x, y, k)$) sau **0** (niciun spike).

#### `SumPooling` — Reducerea Dimensionalității Spațiale

`SumPooling(2, 2)` aplică o fereastră de $2 \times 2$ pe dimensiunile spațiale, **sumând** valorile spike-urilor din fiecare bloc:

$$\text{out}(x', y', c, k) = \sum_{i=0}^{1} \sum_{j=0}^{1} \text{in}(2x'+i,\; 2y'+j,\; c,\; k)$$

**Transformare**: $[56 \times 76 \times 64 \times 4] \rightarrow [28 \times 38 \times 64 \times 4]$

Spre deosebire de Max Pooling (care ia doar valoarea maximă), Sum Pooling **acumulează** activitatea din bloc — o regiune cu 4 spike-uri va avea valoarea 4, iar una cu un singur spike va avea valoarea 1. Aceasta păstrează informația despre **densitatea activării** (câți neuroni au fost activi într-o zonă), nu doar dacă cel puțin un neuron a fost activ.

#### `FeatureScaling` — Normalizarea Feature Maps

`FeatureScaling` normalizează fiecare feature map per-sample la intervalul $[0, 1]$:

$$\hat{x}_i = \frac{x_i - x_{min}}{x_{max} - x_{min} + \epsilon}$$

Această normalizare permite SVM-ului să compare feature maps de sample-uri diferite pe aceeași scară, indiferent de activitatea totală a rețelei (un sample cu o mișcare viguroasă va genera mai multe spike-uri decât unul cu mișcare lentă, dar după normalizare ambele vor avea aceeași amplitudine maximă).

#### Analiza: `Activity` și `Coherence`

- **Activity**: Calculează statistici de activare (sparsity, mean activation, procent de neuroni morți). Un sparsity $\approx 0.24$ indică 24% din pozițiile spațio-temporale au cel puțin un spike — o rată sănătoasă pentru SNN-uri, indicând că rețeaua procesează selectiv.
- **Coherence**: Măsoară **similaritatea cosinus** între ponderile filtrelor, detectând filtre redundante. Un scor de coerență scăzut (Q3 $\approx 0$) indică diversitate excelentă — fiecare filtru a învățat un pattern diferit.

#### `SVM` — Clasificatorul Supervizat Final

Vectorul de feature maps post-pooling $[28 \times 38 \times 64 \times 4]$ este **aplatizat** (flatten) într-un vector unidimensional de $28 \times 38 \times 64 \times 4 = 272{,}384$ elemente, care devine input-ul pentru un clasificator **Support Vector Machine (SVM)** cu kernel liniar.

SVM-ul este antrenat **supervizat** (cu etichete) pe vectorii de feature maps extrași din setul de train, apoi evaluarea se face pe setul de test. Hiperplanele SVM separă cele 6 clase de acțiuni în spațiul de dimensiune înaltă al feature-urilor.

| Strat | Parametri | Input → Output |
|-------|-----------|----------------|
| `TimeObjectiveOutput` | $t_{obj} = 0.75$ | $[56 \times 76 \times 64 \times 4]$ |
| `SumPooling` | $2 \times 2$ | $[56 \times 76 \times 64 \times 4]$ → $[28 \times 38 \times 64 \times 4]$ |
| `FeatureScaling` | — | Normalizare per-sample la $[0, 1]$ |
| `Activity` | — | Statistici: sparsity, active units |
| `Coherence` | — | Diversitatea filtrelor (cosine similarity) |
| **SVM** | Liniar | Clasificare pe 6 clase: boxing, clapping, waving, jogging, running, walking |

**[* FIGURA 12: O diagramă arhitecturală a rețelei complete: Input Video (80×60) → OnOff Filter → LatencyCoding → Conv1 (HOGSampler3D, 64 filtre 5×5×2) → TimeObjectiveOutput → SumPooling 2×2 → FeatureScaling → SVM. Arată dimensiunile tensoriale la fiecare etapă. *]**

### 7.5. Motorul de Execuție: `SparseIntermediateExecution`

Experimentul este orchestrat prin clasa `SparseIntermediateExecution`, care implementează ciclul complet de antrenare/testare:

```
experiment.run(10000):
    1. Citește train dataset (VideoKTH_3D): ~594 videos × 4 samples = ~2376 sample-uri
    2. Preprocesare: OnOff → MaxScaling → LatencyCoding pe fiecare sample
    3. Conv1 Training: 100 epoci × 2376 sample-uri = ~237,600 iterații STDP
    4. Conv1 Testing: propagare forward pe toate sample-urile de train + test
    5. Postprocessing: TimeObjectiveOutput → SumPooling → FeatureScaling
    6. Analize: Activity, Coherence, SVM (clasificare supervizată)
```

Varianta `Sparse` a motorului de execuție optimizează memoria prin procesarea **streaming** a sample-urilor — fiecare sample este procesat și eventual scris pe disc individual, fără a acumula toate sample-urile procesate în memorie RAM simultan.

---

## 8. Vizualizarea Ponderilor și a Hărților de Trăsături

Analiza vizuală a ponderilor sinaptice (weights) și a hărților de trăsături (feature maps) este esențială pentru a valida că rețeaua CSNN a învățat reprezentări semnificative ale mișcării umane. Două scripturi Python, create de autor, automatizează această vizualizare.

### 8.1. Scriptul `draw_kth_weights.py` — Vizualizarea Ponderilor Sinaptice

Scriptul `src/tool/draw_kth_weights.py` citește fișierele JSON de ponderi salvate de simulator la finalul antrenamentului STDP și le vizualizează ca heatmap-uri individuale per filtru, canal și pas temporal.

**Funcționare:**
1. **Detectarea automată a experimentului**: Scriptul caută directorul `Weights/` din `csnn-simulator-build/` și identifică cel mai recent experiment `kth_*` (sau acceptă un nume explicit ca argument).
2. **Parsarea fișierelor JSON**: Fiecare strat convoluțional generează un fișier JSON (ex. `conv1.json`) conținând ponderile sub formă de array 1D, împreună cu dimensiunile tensoriale (`dim_0` ... `dim_4`). Scriptul reconstruiește tensorul multidimensional:
$$\text{weights} \in \mathbb{R}^{N_{filtre} \times f_w \times f_h \times C_{in} \times f_t}$$
   Pentru conv1: $64 \times 5 \times 5 \times 2 \times 2$ = 6400 valori.
3. **Normalizarea min-max per imagine**: Fiecare sub-matrice de $5 \times 5$ este normalizată individual:
$$\hat{w}(i,j) = \frac{w(i,j) - w_{min}}{w_{max} - w_{min}}$$
4. **Generarea imaginilor**: Pentru fiecare combinație (filtru $\times$ canal $\times$ pas temporal), se generează o imagine PNG cu paleta de culori corespunzătoare:
   - **Canalul ON (canal 0)**: Paleta **Blues** (albastru) — reflectă ponderile sinaptice pentru marginile luminoase
   - **Canalul OFF (canal 1)**: Paleta **Reds** (roșu) — reflectă ponderile sinaptice pentru marginile întunecate
   - Fiecare imagine include o bară de culoare (`colorbar`) indicând valorile ponderilor

**Nomenclatura fișierelor de output**: `{layer}_filtru_{N}_ch_{C}_t_{T}.png`
- Exemplu: `conv1_filtru_12_ch_0_t_0.png` = filtrul 12, canalul ON, momentul temporal 0

**Utilizare:**
```bash
cd csnn-simulator-build/
python3 ../src/tool/draw_kth_weights.py              # detectează automat experimentul
python3 ../src/tool/draw_kth_weights.py kth_123       # specifică manual experimentul
```

**Output**: Imaginile sunt salvate în `Weights/{exp_name}_images/{layer}/`.

**Interpretarea vizuală a ponderilor conv1 (2 cadre temporale):**

Pentru conv1 cu `filter_depth = 2`, fiecare filtru produce **2 imagini temporale** (t=0, t=1) pe **2 canale** (ON, OFF) = 4 imagini per filtru, 256 imagini total pentru 64 de filtre.

Comparând cele 2 momente temporale ale aceluiași filtru, se poate observa fenomenul de **shift cinematic** — forma geometrică (de exemplu, conturul gleznei sau al brațului) se deplasează spațial între t=0 și t=1. Acest shift demonstrează că STDP-ul a învățat nu doar **forma statică** a conturului, ci și **direcția și viteza deplasării** (Optical Flow), confirmând că filtrul temporal de adâncime 2 captează informație de mișcare.

**[* FIGURA 14 (IMPORTANTĂ): Inserează 3-4 perechi de imagini ale aceluiași filtru la t=0 și t=1 din `Weights/{exp}_images/conv1/`. Selectează filtre unde shift-ul cinematic este vizibil clar. Sub imagine: "Se observă cum conturul detectat (margine a brațului/piciorului) se deplasează spațial între cele 2 momente temporale, confirmând asimilarea Optical Flow de către STDP." *]**

**[* FIGURA 15: O grilă cu TOATE cele 64 de filtre din conv1 (doar canalul ON, la t=0), aranjate într-o matrice 8×8. Se observă diversitatea formelor: linii orizontale, verticale, diagonale, colțuri, curbe — arsenalul complet de detectoare de margini pe care STDP l-a sculptat nesupervizat. *]**

### 8.2. Scriptul `draw_feature_maps.py` — Vizualizarea Hărților de Activare

Scriptul `src/tool/draw_feature_maps.py` citește fișierele binare de feature maps salvate de `analysis::SaveOutput` și le vizualizează ca heatmap-uri, permițând analiza modului în care rețeaua „vede" fiecare sample.

**Formatul binar al tensorilor**: Simulatorul salvează feature maps-urile într-un format binar propriu, cu:
- **Magic number**: `0x234264FF` (identificator de format)
- **Flag sparse**: 1 bit indicând dacă tensorul este stocat în format dens sau sparse (index + valoare doar pentru elementele nenule)
- **Dimensiuni**: `dim_number` dimensiuni, fiecare pe 2 bytes (`uint16`)
- **Date**: array de `float32` (dens) sau perechi `(uint32 index, float32 valoare)` terminate cu sentinel `0xFFFFFFFF` (sparse)

**Moduri de vizualizare** (adaptate automat la dimensionalitatea tensorului):

| Dimensiuni | Tip tensor | Vizualizare | Exemplu |
|-----------|-----------|-------------|---------|
| 1D ($N$) | FC layer output | **Bar chart** — fiecare bară = activarea unui neuron | fc1: vector de 32 |
| 2D ($H \times W$) | Feature map unică | **Heatmap** (paleta `hot`) | — |
| 3D ($H \times W \times F$) | Multiple feature maps | **Grilă de heatmap-uri** — câte o sub-imagine per filtru | conv2 cu T=1 |
| 4D ($H \times W \times F \times T$) | Feature maps spatio-temporale | **Temporal Max Projection** — $\max_t \text{feature}(x,y,f,t)$ | conv1: [56×76×64×4] |

**Proiecția Temporală Max** (pentru tensori 4D): În loc de a afișa separate cele $T$ momente temporale, se aplică proiecția valorii maxime pe axa temporală:
$$\text{feature\_proj}(x, y, f) = \max_{t \in [0, T)} \text{feature}(x, y, f, t)$$

Aceasta reunește într-o singură imagine spike-urile maxime de-a lungul întregii ferestre temporale — un pixel care a fost activ în oricare din cele $T$ momente va apărea ca activ în proiecție.

**Utilizare:**
```bash
cd csnn-simulator-build/
python3 ../src/tool/draw_feature_maps.py                    # detectează automat
python3 ../src/tool/draw_feature_maps.py --exp-dir ./       # specifică directorul
```

**Output**: `FeatureMaps/{layer}_feature_maps.png`, afișând primele 5 sample-uri.

### 8.3. Interpretarea Vizuală a Rezultatelor

#### Ponderile Conv1 (Weights)

Ponderile sinaptice ale conv1 codifică **detectoare de margini spatio-temporale** pe care STDP-ul le-a auto-organizat din datele de antrenare:
- **Linii orizontale**: Detectează marginile superioară/inferioară ale corpului (umeri, talie)
- **Linii verticale**: Detectează marginile laterale (brațe, picioare în poziție verticală)
- **Linii diagonale**: Detectează unghiurile articulațiilor (cot, genunchi în mișcare)
- **Forme în L / colțuri**: Detectează joncțiunile anatomice complexe (axilă, inghinale)
- **Shift temporal**: Detectează direcția deplasării conturului între cele 2 cadre

#### Feature Maps Conv1 (Activări)

Feature maps-urile conv1 revelă modul în care rețeaua „vede" fiecare sample:
- **Sparsity ridicat** (~75% din pozițiile spațiale sunt zero/negru) — rețeaua activează doar acolo unde găsește match-uri cu filtrele învățate
- **Concentrare pe persoană** — datorită ghidării HOG, activările sunt concentrate pe silueta persoanei, nu pe fundal
- **Diferențe între acțiuni** — walking produce activări distribuite pe vertical (picioare), boxing produce activări concentrate pe zona superioară (brațe/trunchi)

**[* FIGURA 16 (IMPORTANTĂ): Feature Maps din `draw_feature_maps.py` pentru conv1_test — arată 5 sample-uri × 64 filtre cu paleta `hot`. Sub imagine: "Cele 64 de filtre vizualizate prin temporal max projection. Se observă raritatea activărilor (sparsity) și focalizarea pe zona persoanei." *]**

**[* FIGURA 17: Comparație vizuală de feature maps pentru aceleași 3-4 filtre pe câte un sample din fiecare din cele 6 acțiuni. Se observă cum aceleași filtre produc pattern-uri de activare diferite per acțiune — baza pe care SVM-ul construiește separarea claselor. *]**

#### Raritatea Activărilor (Sparsity) și Winner-Takes-All

Un fenomen definitoriu vizibil în hărțile de trăsături este **raritatea extremă a activităților** (sparsity). Majoritatea spațiului receptiv afișează activare zero (culoare neagră), impulsurile concentrându-se exclusiv pe contururile persoanei.

Acest comportament confirmă funcționarea corectă a mecanismului **Winner-Takes-All**: în cadrul unui cluster spațial, doar neuronul (filtrul) cu cel mai ridicat potențial emite spike, inhibându-i pe ceilalți. Consecința: separarea curată a formelor detectate și eficiența energetică masivă — se efectuează calcul doar acolo unde există informație structurală utilă.

**[* FIGURA 18: O secțiune zoom-in din feature maps, selectând 2-3 filtre în care silueta corpului sau brațelor este puternic conturată de zone roșii/galbene pe fundal complet negru. *]**

---

## 9. Rezultate Experimentale și Comparații

**[* REZULTATE: Inserează tabelul cu acuratețea SVM obținută pe conv1 pentru experimentul curent (seed=123, 64 filtre, 5 cadre temporale, HOGSampler3D). *]**

**[* REZULTATE: Matricea de Confuzie (Confusion Matrix) pentru conv1. Comentează pe seama ei: ce perechi de acțiuni se confundă (probabil jogging/running, boxing/handclapping). *]**

**[* REZULTATE: Un bar chart cu acuratețile per clasă (boxing, handclapping, handwaving, jogging, running, walking) la conv1. *]**

**[* COMPARAȚIE: Tabel comparativ între diferite configurații experimentale (HOGSampler3D vs. RandomSampler3D, diferite numere de filtre, diferite temporal_kernel). *]**

**[* COMPARAȚIE: Filtrele (weights) învățate cu HOGSampler3D vs. RandomSampler3D — inserează imagini din `Weights/` pentru ambele variante. *]**

**[* REZULTATE: Graficul evoluției sparsity-ului pe conv1 (analiza Activity). *]**

**[* REZULTATE: Analiza coerenței filtrelor (Coherence) — tabel cu Mean Weights, Coherence Q3, Coherence Max. Interpretează diversitatea filtrelor. *]**

**[* FIGURA IMPORTANTA: Inserează pozele cu Greutățile (Weights) generate de `draw_kth_weights.py` — arată filtrele pe canalul ON (Blues) și OFF (Reds), pe cele 2 momente temporale (t=0, t=1). Explică cum se observă shift-ul cinematic între cele 2 momente (Optical Flow). *]**

**[* FIGURA IMPORTANTA: Feature Maps din `draw_feature_maps.py` pentru conv1, exemplificând modul abstractizat prin care SNN-ul "vede" omul. *]**

**[* COMPARAȚIE: Rezultate pe varianta 160×120 (cu mog2_min_area=720) vs. 80×60 (cu mog2_min_area=180), dacă sunt disponibile. *]**

---

## 9. Clasificatorul Predictiv SVM (Support Vector Machine)

Odată arhitectura SNN antrenată nesupervizat, plasticitatea stratului STDP este suspendată (imobilizând greutățile). Acum, dataset-ului global i se extrage doar semnătura "spiking" din ultimul strat conv al CSNN.

Fiecare videoclip comprimat tridimensional în descărcări electrice devine pură prelucrare vectorială — livrând argumente matematice către un identificator supervizat de tip **Support Vector Machine (SVM)** [21] liniar. SVM-ul construiește hiperplane matematice pentru demarcația deciziilor dintre cele 6 clase de comportament, implementat prin biblioteca libsvm [30].

Clasificarea se face pe baza feature map-urilor extrase prin `TimeObjectiveOutput` urmate de `SumPooling` și `FeatureScaling`, permițând analiza calității reprezentărilor la nivelul de abstractizare al fiecărui strat.

**[* REZULTATE: Tabel final cu acuratețe SVM per configurare experimentală. *]**

---

## 10. Optimizări de Memorie și Performanță

### 10.1. Eliminarea Memory Leak-ului HOG prin Separarea Offline/Runtime

O contribuție tehnică importantă a fost **rezolvarea problemei de Out-of-Memory (OOM Kill)** cauzată de rularea detecției HOG **la runtime** în interiorul buclei STDP.

**Problema originală**: În versiunea anterioară a simulatorului, detectorul HOG OpenCV era apelat **la fiecare iterație STDP** pentru a calcula bounding box-ul persoanei. Cu 100 de epoci × 2376 de sample-uri, aceasta genera peste **237,000 de apeluri** `cv2.HOGDescriptor::detectMultiScale`, fiecare alocând memorie internă pentru piramida de imagini, histogramele de orientare și structurile SVM. Aceste alocări repetate **fragmentau Heap-ul** progresiv, ducând la un consum de memorie crescând care culmina cu OOM Kill pe sisteme cu 16GB RAM (tipic după 40-60 de epoci).

**Soluția**: Mutarea completă a detecției HOG într-un **script Python offline** (`extract_bboxes_kth.py`) care rulează o singură dată, salvând rezultatele în JSON. La runtime, simulatorul C++ citește doar coordonatele numerice din JSON — **fără nicio dependență de OpenCV pentru detecție**, eliminând complet problema de fragmentare a memoriei.

**Impact**: Consumul de memorie la runtime a scăzut de la ~20GB+ (crescând) la ~8GB (constant), permițând execuția completă pe hardware cu 16GB RAM.

### 10.2. Streaming Mode în Procesarea Output-ului

O a doua optimizare de memorie a vizat funcția `_process_output` din `SparseIntermediateExecution`. Problema: funcția crea **copii complete** ale tuturor sample-urilor procesate la rezoluția conv1 ($56 \times 76 \times 64 \times 4 \approx 1{,}090{,}048$ valori per sample) **înainte** de a aplica postprocessing-ul care le reduce dramatic la $28 \times 38 \times 64 \times 4$.

Cu ~2376 sample-uri, acumularea pre-pooling consuma:
$$2376 \times 1{,}090{,}048 \times 4 \text{ bytes} \approx 9.7 \text{ GB}$$

**Soluția implementată** introduce două moduri de procesare:
1. **Streaming mode**: Pentru output-uri fără postprocessing (ex. `SaveOutput`), sample-urile sunt procesate și scrise pe disc **individual**, fără a fi acumulate în memorie.
2. **Inline postprocessing**: Pentru output-uri cu postprocessing (`SumPooling` + `FeatureScaling`), pașii sunt aplicați **per-sample inline** — doar rezultatele post-pooling (de dimensiuni reduse) sunt stocate în memorie.

**Impact**: Reducerea consumului de memorie de peak de la ~36GB la ~12GB, permițând execuția completă cu postprocessing pe sisteme cu 16GB RAM.

**[* FIGURA 13: Un grafic de memorie (RAM usage vs. timp/epoci) comparând versiunea originală (linie crescătoare spre OOM) cu versiunea optimizată (linie plată la ~8-12GB). *]**

---

## 11. Contribuții Originale

### 11.1. Scriptul `extract_bboxes_kth.py` cu Detecție Duală HOG + MOG2

Scriptul de pre-procesare, creat de autor, combină doi algoritmi complementari de viziune artificială printr-o **fuziune per-cadru** cu scorare multi-criterială, netezire EMA și carry temporal. Această abordare duală procesează cu succes **594 din 599 videoclipuri** (99.2%), comparativ cu sub 90% într-o configurație doar HOG.

### 11.2. Crearea Clasei `VideoKTH_3D`

Clasa de dataset, creată de la zero, înlocuiește eșantionarea secvențială/aleatorie cu **extragere ghidată pe baza datelor HOG pre-calculate**, construind simultan un **sistem de mapare statică** (`sample_index → (video_key, group_idx)`) care permite sampler-ului să acceseze bounding box-urile corecte fără dependențe directe de dataset.

### 11.3. Arhitectura Sampler (Strategy Pattern) și `HOGSampler3D`

Logica de eșantionare a fost **extrasă din clasa de convoluție** într-o ierarhie de clase `Sampler` independente. Implementarea `HOGSampler3D` folosește un **sistem de cache** cu **fallback temporal pe 3 niveluri**, focalizând eșantionarea STDP pe zona persoanei fără nicio dependență de OpenCV la runtime.

### 11.4. `ConvolutionSampler3D`

Wrapper semantic peste `Convolution3D` care permite experimentele sampler-driven să fie identificabile vizual în cod, fără duplicare de logică.

### 11.5. Optimizări de Memorie

- **Separarea Offline/Runtime**: Eliminarea memory leak-ului prin mutarea detecției HOG din runtime (buclă STDP) în scriptul Python offline.
- **Streaming Mode**: Procesarea și scrierea sample-urilor individual pe disc, fără acumularea lor pre-pooling în memorie RAM, reducând consumul de la ~36GB la ~12GB.

**[* REZULTATE FINALE: Tabel rezumativ cu toate experimentele rulate, parametrii, și acuratețile obținute. *]**

**[* COMPARAȚIE FINALĂ: Grafic bar chart sau line plot cu evoluția performanței pe parcursul diferitelor versiuni ale pipeline-ului (de la random sampling la HOG-guided, de la HOG_Convolution3D hardcoded la Sampler Pattern). *]**

---

## 12. Limitări ale Abordării și Direcții de Îmbunătățire

Orice sistem experimental trebuie evaluat nu doar prin performanțele obținute, ci și prin **constrângerile intrinseci** care îi definesc plafonul de performanță. Această secțiune identifică și analizează sistematic limitările pipeline-ului CSNN prezentat, explicând cauzele lor tehnice și propunând posibile direcții de ameliorare.

### 12.1. Discrepanța de Acuratețe SNN vs. CNN pe KTH

Cea mai vizibilă limitare a abordării este **diferența de acuratețe** față de rețelele neuronale convoluționale clasice (CNN). Pe KTH, arhitecturi supervizate precum C3D, ResNet-3D sau Two-Stream Networks ating acuratețe de **92–98%**, în timp ce pipeline-ul CSNN cu un singur strat convoluțional și antrenare nesupervizată STDP obține valori semnificativ mai mici.

**Cauzele tehnice ale acestei discrepanțe:**

1. **Antrenare nesupervizată vs. supervizată**: STDP optimizează corelații temporale locale între spike-uri de intrare și ieșire, **fără nicio cunoaștere a etichetelor de clasă**. Prin contrast, backpropagation-ul într-un CNN minimizează direct eroarea de clasificare — o funcție obiectiv mult mai informativă. Formal:

   - **STDP**: Ponderile se ajustează pe baza coincidențelor temporale:
   $$\Delta w \propto \begin{cases} A^+ \cdot e^{-\Delta t / \tau^+} & \text{dacă } \Delta t > 0 \text{ (pre-synaptic → post-synaptic)} \\ -A^- \cdot e^{\Delta t / \tau^-} & \text{dacă } \Delta t < 0 \end{cases}$$
   
   - **Backpropagation**: Ponderile se ajustează pe baza gradientului erorii de clasificare:
   $$\Delta w = -\eta \cdot \frac{\partial \mathcal{L}(y, \hat{y})}{\partial w}$$
   
   STDP descoperă **structuri statistice** din date (margini, texturi, mișcare); backpropagation-ul descoperă **trăsături discriminative** între clase. Consecința: STDP poate converge spre filtre redundante sau irelevante pentru clasificare.

2. **Adâncimea rețelei**: Pipeline-ul actual folosește **un singur strat convoluțional** (conv1), care extrage trăsături de nivel scăzut (margini, colțuri, direcții de mișcare). CNN-urile performante pe acțiuni video au 10–100+ de straturi, formând **ierarhii de abstractizare**:
   
   | Nivel | CNN (supervizat) | CSNN actual (1 strat) |
   |-------|------------------|-----------------------|
   | Margini, texturi | ✅ Conv1-2 | ✅ Conv1 |
   | Părți de corp (braț, picior) | ✅ Conv3-5 | ❌ — |
   | Pose complete (ridicat, aplecat) | ✅ Conv6-8 | ❌ — |
   | Secvențe de acțiuni | ✅ FC layers | ❌ — |
   
   Cu un singur strat, rețeaua delegă **întreaga complexitate de abstractizare** clasificatorului SVM, care primește doar hărți de margini — nu reprezentări semantice.

3. **Absența augmentării datelor**: CNN-urile standard beneficiază de augmentări masive (crop aleator, flip, jitter de culoare, mixup), multiplicând diversitatea aparentă a setului de antrenare. Pipeline-ul CSNN nu aplică augmentare, limitând generalizarea la variațiile prezente nativ în dataset.

**[* TABEL COMPARATIV: Rezultatele CSNN-ului (acuratețe per clasă și medie) vs. rezultate publicate din literatura de specialitate pentru KTH — C3D (Tran et al., 2015), Two-Stream (Simonyan & Zisserman, 2014), LRCN (Donahue et al., 2015), și alte SNN-uri publicate. Surse: papers originale. *]**

### 12.2. Confuzia între Acțiuni Cinematric Similare

O limitare recurentă pe KTH este **confuzia sistematică** între perechile de acțiuni cu cinematică similară:

#### 12.2.1. Jogging vs. Running

Aceste două acțiuni implică **aceeași secvență de mișcări** ale membrelor inferioare (alternare picioare, balans brațe), diferind doar prin **amplitudinea și frecvența mișcării**:
- **Running**: Pași mai lungi, faza de zbor mai extinsă, frecvență mai mare
- **Jogging**: Pași mai scurți, contact mai lung cu solul, frecvență mai mică

La rezoluția de 80×60 pixeli cu fereastră temporală de 5 cadre, aceste diferențe subtile de amplitudine și frecvență sunt greu de distins:
$$\text{Jogging}: v_{deplasare} \approx 2\text{-}4 \text{ px/cadru}, \quad \text{Running}: v_{deplasare} \approx 4\text{-}7 \text{ px/cadru}$$

Cu filtre convoluționale de 5×5 pixeli spațial și 2 cadre temporal, diferența de 2-3 pixeli/cadru în viteza de deplasare este la **limita rezoluției de detecție** a unui singur filtru. Consecința: multe filtre conv1 emit pattern-uri de activare aproape identice pentru cele două acțiuni, iar SVM-ul nu găsește un hiperplan de separare clar.

#### 12.2.2. Boxing vs. Handclapping

Ambele acțiuni implică mișcări rapide ale brațelor în zona superioară a corpului. La nivel de margini spatio-temporale (output conv1), profilul de mișcare este similar — diferiră în principal prin **simetria mișcării** (boxing: brațe alternante; handclapping: brațe simultane), o distincție dificil de capturat cu filtre locale mici.

#### 12.2.3. Handwaving vs. Handclapping

Ambele acțiuni implică mișcări ale brațelor deasupra umerilor. Handwaving are o componentă laterală mai pronunțată, dar la rezoluție scăzută, contururile de margini se pot confunda.

**Impact pe matricea de confuzie**: Aceste confuzii se manifestă ca valori nenule **off-diagonal** concentrate pe perechile (jogging, running) și (boxing, handclapping). Acuratețea per clasă pentru jogging și running este de obicei cu **10-20 puncte procentuale** sub media generală.

**[* FIGURA 19 (IMPORTANTĂ): Matricea de confuzie (confusion matrix) din experimentul curent. Evidențiază cu săgeți/cercuri perechile confuze (jogging↔running, boxing↔handclapping). Sub imagine: "Se observă concentrarea erorilor pe perechile de acțiuni cinematric similare, confirmând limitarea rezoluției spatio-temporale a unui singur strat convoluțional." *]**

**[* FIGURA 20: Un montaj de 4 cadre consecutive din jogging și 4 cadre consecutive din running, alăturate. Sub imagine: "La rezoluția 80×60, diferențele vizuale sunt subtile — silueta, postura și secvența de mișcări sunt aproape identice." *]**

### 12.3. Problemele de Variabilitate a Scalei (Scenariul s2)

Datasetul KTH include 4 scenarii de filmare, dintre care **scenariul s2** prezintă camera la o **distanță mai mare** de subiect. Consecințele sunt:

1. **Persoana ocupă o fracțiune mai mică a cadrului**: La 80×60 pixeli, silueta în s2 poate ocupa doar 15-20% din cadru (vs. 30-50% în s1/s3/s4)
2. **Bounding box-urile HOG sunt mai mici**: Detectorul HOG la scala redusă 3× are dificultăți în detectarea persoanelor la distanță mare, crescând rata de fallback la detecția MOG2
3. **Filtrele 5×5 acoperă o proporție mai mare din corp**: Într-un cadru s2, filtrul 5×5 poate acoperi jumătate din persoană, pierzând informația de structură locală

**Consecința**: Performanța per-scenariu este neuniformă — acuratețea pe s2 este consistent mai scăzută decât pe s1/s3/s4.

**[* TABEL: Acuratețea SVM descompusă per scenariu (s1, s2, s3, s4), evidențiind degradarea pe s2. *]**

**[* FIGURA 21: Două cadre comparative — același subiect, aceeași acțiune, în s1 (apropiat) vs. s2 (la distanță), ambele la rezoluția 80×60 pixeli. Sub imagine: "Variabilitatea scalei între scenarii afectează semnificativ raportul semnal/zgomot al bounding box-urilor detectate." *]**

### 12.4. Problema Sparsity-ului la Straturi Profunde

Un fenomen observat în experimentele multi-strat (și documentat în literatura SNN) este **creșterea progresivă a sparsity-ului** pe măsură ce se adaugă straturi convoluționale:

$$\text{Sparsity}(l) = 1 - \frac{|\{s_{ijk} > 0\}|}{H_l \times W_l \times F_l}$$

La conv1, sparsity-ul tipic este de ~60-75% (valoare sănătoasă — suficiente activări pentru antrenarea stratului următor). Însă la un ipotetic conv2 antrenat pe output-ul conv1, sparsity-ul poate crește la ~90-95%, iar la conv3 poate atinge **~99-100%** (moarte neuronală completă).

**Cauzele:**
- **Rarefacerea cascadată**: Fiecare strat introduce un WTA care elimină competitori, apoi stratul următor primește un input deja rareficat
- **Praguri homeostatice cumulative**: Dacă pragul $\theta$ crește prea agresiv la niveluri superioare, neuronii nu mai au suficiente spike-uri de intrare pentru a atinge pragul
- **Pierderea informației temporale**: SumPooling între straturi reduce dimensiunea temporală; la conv2 cu input deja pooled temporal, dimensionalitatea temporală rămasă poate fi insuficientă

**Consecința practică**: Un sparsity de 1.0 la un strat semnifică **zero spike-uri de output** — stratul respectiv nu transmite nicio informație utilă, iar clasificatorul SVM primește un vector de zerouri. Aceasta este una dintre motivațiile pentru care pipeline-ul actual folosește un singur strat convoluțional, nu o stivă profundă.

**[* FIGURA 22: Un grafic bar chart cu Sparsity per strat (conv1, conv2, conv3) din experimentele multi-strat, arătând creșterea progresivă spre 1.0. *]**

### 12.5. Limitarea Ferestrei Temporale

Fereastra temporală actuală de **5 cadre** cu `frame_gap=0` acoperă un interval real de:
$$T_{real} = \frac{5 \text{ cadre}}{25 \text{ fps}} = 0.2 \text{ secunde}$$

Un ciclu complet de mers (walking) durează ~1 secundă (25 cadre), iar un ciclu de alergare (running) ~0.5 secunde (12-13 cadre). Cu o fereastră de 0.2 secunde, rețeaua observă doar **20-40% dintr-un ciclu de mișcare**, capturând un fragment de tranziție, nu secvența completă.

| Acțiune | Durata ciclu | Cadre/ciclu (25fps) | Procent captat (5 cadre) |
|---------|-------------|---------------------|--------------------------|
| Walking | ~1.0 s | ~25 | ~20% |
| Running | ~0.5 s | ~12 | ~42% |
| Jogging | ~0.7 s | ~17 | ~29% |
| Boxing | ~0.8 s | ~20 | ~25% |
| Handclapping | ~0.6 s | ~15 | ~33% |
| Handwaving | ~0.8 s | ~20 | ~25% |

Creșterea ferestrei (ex. la 10 sau 15 cadre) ar capta cicluri complete, dar ar crește exponențial cerințele de memorie (dimensiunea filtrelor temporale) și ar necesita mai multe sample-uri de antrenare. Un compromis ar fi utilizarea `frame_gap > 0` pentru a acoperi un interval temporal mai lung cu același număr de cadre.

### 12.6. Sensibilitatea la Hiperparametri

Pipeline-ul CSNN conține un număr semnificativ de hiperparametri, iar performanța este sensibilă la configurarea lor:

| Hiperparametru | Valoare curentă | Impact la variație |
|---------------|----------------|-------------------|
| $\eta_w$ (learning rate STDP) | 0.1 | Prea mare → ponderi saturate; prea mică → convergență lentă |
| $\theta_0$ (prag inițial) | $\mathcal{N}(8.0, 0.1)$ | Prea mare → neuroni silențioși; prea mic → sparsity insuficient |
| $\eta_\theta$ (lr threshold) | 1.0 | Controlează viteza de adaptare homeostatică |
| $t_{obj}$ (fereastră temporală output) | 0.75 | Filtrează spike-urile tardive; prea mic → pierdere informație |
| $\alpha$ (annealing rate) | 0.95 | Controlează decelerarea STDP peste epoci |
| Nr. filtre | 64 | Prea puține → sub-reprezentare; prea multe → redundanță |
| Dimensiune filtru | 5×5×2 | Trade-off receptive field vs. specificitate |
| Nr. epoci | 150 | Insuficiente → subantrenare; prea multe → saturare |
| seed | 40 | Inițializare aleatorie — rezultatele variază cu ±2-3% per seed |

Spre deosebire de CNN-uri unde tehnici precum Adam, learning rate scheduling și batch normalization reduc sensibilitatea la hiperparametri, STDP-ul nu beneficiază de aceste instrumente stabilizatoare.

**[* TABEL: Rezultate dintr-un sweep de hiperparametri — variind lr_w, threshold, număr de filtre. Arată cum variază acuratețea SVM. *]**

### 12.7. Limitări ale Sistemului de Detecție a Persoanei

Deși fuziunea HOG+MOG2 procesează 594/599 videoclipuri cu succes, sistemul de bounding boxes are limitări inerente:

1. **5 videoclipuri nedetectate**: Cele 5 videoclipuri ratate (~0.8%) conțin condiții de filmare atipice (iluminare extremă, persoana parțial ascunsă de obiecte, sau mișcare minimală care nu activează MOG2). Aceste videoclipuri sunt excluse complet din antrenare/testare.

2. **Calitatea variabilă a bbox-urilor**: Chiar și pentru videoclipurile detectate cu succes, fuziunea HOG+MOG2 produce bbox-uri de calitate neuniformă:
   - **Detecții parțiale**: Bbox-ul acoperă doar trunchiul, nu întregul corp (frecvent pentru persoane la marginea cadrului)
   - **Detecții excesiv de largi**: MOG2 activat de mișcări de fundal (arbori, drapele) produce bbox-uri dilatate
   - **Fluctuații temporale**: Deși netezirea EMA ($\alpha = 0.65$) reduce jitter-ul, tranzițiile între detecția HOG și MOG2 pot produce salturi

3. **Dependența de calitatea preprocessării**: Orice eroare în bounding box se propagă cascadat: bbox incorect → sampling greșit → STDP antrenează pe background → filtre de zgomot → acuratețe degradată.

### 12.8. Direcții de Îmbunătățire Viitoare

Pe baza limitărilor identificate, se conturează următoarele direcții de dezvoltare:

1. **Arhitectură multi-strat cu anti-sparsity**: Introducerea de straturi conv2 și conv3 cu mecanisme de reglare a sparsity-ului (ex. prag adaptiv cu target firing rate mai agresiv, sau lateral excitation între filtre)

2. **Fereastră temporală adaptivă**: Ajustarea automată a `frame_gap` sau `video_frames` în funcție de viteza de mișcare detectată în sample-ul curent

3. **Sampler hibrid multi-strat**: Straturi superioare (conv2+) ar putea folosi un sampler ghidat de activările stratului anterior, nu de bbox-urile HOG — concentrând antrenarea pe zonele deja validate ca informative

4. **Augmentare la nivel de spike trains**: Augmentarea datelor nu la nivel de pixeli (ca în CNN-uri), ci la nivel de trenuri de spike-uri — adăugând jitter temporal, permutări de ordine, sau noise Poisson

5. **Detecție robustă cu rețele neuronale profunde**: Înlocuirea detectorului HOG+MOG2 cu un detector bazat pe deep learning (YOLO, Faster R-CNN) pentru bbox-uri mai precise și consistente

6. **Evaluare cross-dataset**: Testarea pipeline-ului pe alte dataset-uri de acțiuni (UCF-101, HMDB-51) pentru a evalua generalizarea dincolo de KTH

**[* FIGURA 23: O diagramă conceptuală a arhitecturii viitoare propuse — conv1 (HOGSampler) → conv2 (ActivationSampler) → conv3 → FC → SVM, cu indicații de feedback între straturi. *]**

---

## Bibliografie

[1] R. Poppe, "A survey on vision-based human action recognition," *Image and Vision Computing*, vol. 28, no. 6, pp. 976–990, 2010.

[2] S. Herath, M. Harandi, and F. Porikli, "Going deeper into action recognition: A survey," *Image and Vision Computing*, vol. 60, pp. 4–21, 2017.

[3] D. Tran, L. Bourdev, R. Fergus, L. Torresani, and M. Palber, "Learning spatiotemporal features with 3D convolutional networks," in *Proc. IEEE ICCV*, pp. 4489–4497, 2015.

[4] K. Simonyan and A. Zisserman, "Two-stream convolutional networks for action recognition in videos," in *Proc. NeurIPS*, pp. 568–576, 2014.

[5] J. Carreira and A. Zisserman, "Quo vadis, action recognition? A new model and the Kinetics dataset," in *Proc. IEEE CVPR*, pp. 6299–6308, 2017.

[6] C. Feichtenhofer, H. Fan, J. Malik, and K. He, "SlowFast networks for video recognition," in *Proc. IEEE ICCV*, pp. 6202–6211, 2019.

[7] W. Maass, "Networks of spiking neurons: The third generation of neural network models," *Neural Networks*, vol. 10, no. 9, pp. 1659–1671, 1997.

[8] A. Tavanaei, M. Ghodrati, S. R. Kheradpisheh, T. Masquelier, and A. Maida, "Deep learning in spiking neural networks," *Neural Networks*, vol. 111, pp. 47–63, 2019.

[9] S. Thorpe, D. Fize, and C. Marlot, "Speed of processing in the human visual system," *Nature*, vol. 381, pp. 520–522, 1996.

[10] M. Davies et al., "Advancing neuromorphic computing with Loihi: A survey of results and outlook," *Proceedings of the IEEE*, vol. 109, no. 5, pp. 911–934, 2021.

[11] M. Davies et al., "Loihi: A neuromorphic manycore processor with on-chip learning," *IEEE Micro*, vol. 38, no. 1, pp. 82–99, 2018.

[12] P. A. Merolla et al., "A million spiking-neuron integrated circuit with a scalable communication network and interface," *Science*, vol. 345, no. 6197, pp. 668–673, 2014.

[13] S. B. Furber, F. Galluppi, S. Temple, and L. A. Plana, "The SpiNNaker project," *Proceedings of the IEEE*, vol. 102, no. 5, pp. 652–665, 2014.

[14] G. Gallego et al., "Event-based vision: A survey," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 44, no. 1, pp. 154–180, 2022.

[15] G.-Q. Bi and M.-M. Poo, "Synaptic modifications in cultured hippocampal neurons: Dependence on spike timing, synaptic strength, and postsynaptic cell type," *Journal of Neuroscience*, vol. 18, no. 24, pp. 10464–10472, 1998.

[16] S. Song, K. D. Miller, and L. F. Abbott, "Competitive Hebbian learning through spike-timing-dependent synaptic plasticity," *Nature Neuroscience*, vol. 3, no. 9, pp. 919–926, 2000.

[17] T. Masquelier and S. Thorpe, "Unsupervised learning of visual features through spike timing dependent plasticity," *PLoS Computational Biology*, vol. 3, no. 2, e31, 2007.

[18] C. Schuldt, I. Laptev, and B. Caputo, "Recognizing human actions: A local SVM approach," in *Proc. IEEE ICPR*, vol. 3, pp. 32–36, 2004.

[19] N. Dalal and B. Triggs, "Histograms of oriented gradients for human detection," in *Proc. IEEE CVPR*, vol. 1, pp. 886–893, 2005.

[20] Z. Zivkovic, "Improved adaptive Gaussian mixture model for background subtraction," in *Proc. IEEE ICPR*, vol. 2, pp. 28–31, 2004.

[21] C. Cortes and V. Vapnik, "Support-vector networks," *Machine Learning*, vol. 20, no. 3, pp. 273–297, 1995.

[22] I. Laptev, "On space-time interest points," *International Journal of Computer Vision*, vol. 64, no. 2–3, pp. 107–123, 2005.

[23] H. Wang, A. Kläser, C. Schmid, and C.-L. Liu, "Action recognition by dense trajectories," in *Proc. IEEE CVPR*, pp. 3169–3176, 2011.

[24] J. Sivic and A. Zisserman, "Video Google: A text retrieval approach to object matching in videos," in *Proc. IEEE ICCV*, pp. 1470–1477, 2003.

[25] S. R. Kheradpisheh, M. Ganjtabesh, S. J. Thorpe, and T. Masquelier, "STDP-based spiking deep convolutional neural networks for object recognition," *Neural Networks*, vol. 99, pp. 56–67, 2018.

[26] M. Mozafari, M. Ganjtabesh, A. Nowzari-Dalini, and T. Masquelier, "SpykeTorch: Efficient simulation of convolutional spiking neural networks with at most one spike per neuron," *Frontiers in Neuroscience*, vol. 13, article 625, 2019.

[27] P. Falez, P. Music, D. Ber, F. Cemesse, and T. Music, "Multi-layered spiking neural network with target timestamp threshold adaptation and STDP," in *Proc. IJCNN*, pp. 1–8, 2019.

[28] C. M. Parameshwara et al., "SpikeMS: Deep spiking neural network for motion segmentation," in *Proc. IEEE/RSJ IROS*, pp. 3414–3420, 2021.

[29] W. Fang et al., "SpikingJelly: An open-source machine learning infrastructure platform for spike-based intelligence," *arXiv preprint arXiv:2310.16620*, 2023.

[30] C.-C. Chang and C.-J. Lin, "LIBSVM: A library for support vector machines," *ACM Transactions on Intelligent Systems and Technology*, vol. 2, no. 3, article 27, 2011.

[31] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, "Learning representations by back-propagating errors," *Nature*, vol. 323, pp. 533–536, 1986.

[32] Y. LeCun, Y. Bengio, and G. Hinton, "Deep learning," *Nature*, vol. 521, pp. 436–444, 2015.

[33] P. U. Diehl and M. Cook, "Unsupervised learning of digit recognition using spike-timing-dependent plasticity," *Frontiers in Computational Neuroscience*, vol. 9, article 99, 2015.

[34] D. H. Hubel and T. N. Wiesel, "Receptive fields, binocular interaction and functional architecture in the cat's striate cortex," *Journal of Physiology*, vol. 160, no. 1, pp. 106–154, 1962.

[35] K. Fukushima, "Neocognitron: A self-organizing neural network model for a mechanism of pattern recognition unaffected by shift in position," *Biological Cybernetics*, vol. 36, no. 4, pp. 193–202, 1980.

[36] B. D. Lucas and T. Kanade, "An iterative image registration technique with an application to stereo vision," in *Proc. DARPA IUW*, pp. 121–130, 1981.

[37] D. G. Lowe, "Distinctive image features from scale-invariant keypoints," *International Journal of Computer Vision*, vol. 60, no. 2, pp. 91–110, 2004.

[38] G. Bradski, "The OpenCV Library," *Dr. Dobb's Journal of Software Tools*, 2000.

[39] D. Marr and E. Hildreth, "Theory of edge detection," *Proceedings of the Royal Society of London B*, vol. 207, no. 1167, pp. 187–217, 1980.