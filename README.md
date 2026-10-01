# Apex ISR — Intelligence, Surveillance, Reconnaissance

Sensor fusion, multi-INT fusion, track management with Kalman/EKF filters.

## Architecture

```mermaid
flowchart TD
    subgraph Sensors["Multi-Domain Sensors"]
        RADAR[Radar]
        EOIR[EO/IR]
        SIGINT[SIGINT]
        HUMINT[HUMINT]
        GEOINT[GEOINT]
        OSINT[OSINT]
        MASINT[MASINT]
        CYBINT[CYBINT]
    end

    subgraph FusionLayer["Sensor Fusion Layer"]
        KF[Linear Kalman Filter\nCV/CA/CT Models]
        EKF[Extended Kalman Filter\nNon-linear]
        AKF[Adaptive Kalman Filter\nRobust Estimation]
        PF[Particle Filter\nNon-Gaussian]
        IMM[Interacting Multiple Model]
        UD[UD Factorization\nNumerical Stability]
    end

    subgraph AssociationLayer["Data Association"]
        GNN[Global Nearest Neighbor\nHungarian Algorithm]
        JPDA[Joint Probabilistic\nData Association]
        MHT[Multiple Hypothesis\nTracking]
    end

    subgraph TrackLayer["Track Management"]
        TI[Track Initiation\nM-of-N]
        TC[Track Confirmation]
        TM[Track Maintenance\nPrediction]
        TD[Track Deletion]
        TQ[Track Quality Scoring]
        TL[Track Lifecycle]
    end

    subgraph INTLayer["Multi-INT Fusion"]
        ER[Entity Resolution\nCross-INT Matching]
        EI[Entity Disambiguation]
        IG[Identity Management]
        CI[Cross-INT Correlation]
    end

    subgraph ISRLayer["ISR Pipeline"]
        COL[Collection Management\nSensor Tasking]
        PROC[Processing\nFeature Extraction]
        EXP[Exploitation\nPattern Recognition]
        DIS[Dissemination\nSTANAG 4607]
    end

    subgraph ThreatLayer["Threat Assessment"]
        TA[Threat Scoring]
        IP[Intent Prediction]
        BA[Behavioral Analysis]
        AD[Anomaly Detection]
    end

    RADAR --> KF
    EOIR --> KF
    SIGINT --> EKF
    HUMINT --> ER
    GEOINT --> ER
    OSINT --> ER
    MASINT --> PF
    CYBINT --> ER

    KF --> GNN
    EKF --> GNN
    AKF --> JPDA
    PF --> MHT
    IMM --> GNN
    UD --> KF

    GNN --> TI
    JPDA --> TI
    MHT --> TI
    TI --> TC --> TM --> TD
    TM --> TQ
    TM --> TL

    ER --> IG
    EI --> IG
    CI --> IG

    IG --> COL
    COL --> PROC --> EXP --> DIS

    EXP --> TA
    TA --> IP
    IP --> BA
    BA --> AD
```

## Sensor Fusion Pipeline

```mermaid
flowchart LR
    RAW[Raw Sensor Data] --> PRE[Preprocessing\nAlignment/Calibration]
    PRE --> KF[Kalman Filter\nState Estimation]
    KF --> EKF{Non-linear?}
    EKF -->|Yes| NL[Non-linear Update]
    EKF -->|No| LIN[Linear Update]
    NL --> PF{Non-Gaussian?}
    LIN --> PF
    PF -->|Yes| SMP[Particle Filter]
    PF -->|No| GAU[Gaussian Approximation]
    SMP --> FUSED[Fused State Estimate]
    GAU --> FUSED
    FUSED --> TRACK[Track Output]
```

## Track Management Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initiation: M-of-N Detection
    Initiation --> Confirmation: M detections in N scans
    Confirmation --> Maintenance: Track Established
    Maintenance --> Maintenance: Update Position/Velocity
    Maintenance --> Deletion: Missed Detections > Threshold
    Maintenance --> Split: Multiple Clusters Detected
    Maintenance --> Merge: Single Cluster Detected
    Split --> Maintenance: New Track Created
    Merge --> Maintenance: Track Consolidated
    Deletion --> [*]: Track Terminated
```

## Multi-INT Fusion Architecture

```mermaid
flowchart TD
    subgraph INTs["INT Disciplines"]
        HUMINT[HUMINT\nHuman Intelligence]
        SIGINT[SIGINT\nSignals Intelligence]
        GEOINT[GEOINT\nGeospatial Intelligence]
        OSINT[OSINT\nOpen Source Intelligence]
        MASINT[MASINT\nMeasurement & Signature]
        CYBINT[CYBINT\nCyber Intelligence]
        FININT[FININT\nFinancial Intelligence]
    end

    subgraph Fusion["Cross-INT Fusion"]
        ER[Entity Resolution\nConfidence Scoring]
        EI[Entity Disambiguation\nIdentity Resolution]
        CR[Cross-INT Correlation\nPattern Detection]
        IG[Identity Graph\nKnowledge Graph]
    end

    subgraph Output["Intelligence Products"]
        TA[Threat Assessment]
        IP[Intent Prediction]
        BA[Behavioral Analysis]
        AL[Alerts\nPriority Routing]
    end

    HUMINT --> ER
    SIGINT --> ER
    GEOINT --> ER
    OSINT --> ER
    MASINT --> CR
    CYBINT --> CR
    FININT --> CR

    ER --> EI --> IG
    CR --> IG

    IG --> TA
    IG --> IP
    IG --> BA
    IG --> AL
```

## Benchmark Comparisons

| Feature | Apex ISR | Palantir Gotham | Anduril Lattice | TAK/ATAK |
|---------|---------|-----------------|-----------------|----------|
| Sensor Fusion | Kalman/EKF/IMM/Particle | Proprietary | Proprietary | Basic |
| Data Association | GNN/JPDA/MHT | ❌ | ❌ | ❌ |
| Track Management | M-of-N lifecycle | ✅ | ✅ | ✅ |
| Multi-INT Fusion | 7 INT disciplines | ✅ | ❌ | ❌ |
| NATO STANAG | 4607 | ❌ | ❌ | ✅ |
| Open Source | AGPL-3.0 | ❌ | ❌ | ❌ |
| Real-time Processing | ✅ | ✅ | ✅ | ✅ |
| Entity Resolution | Cross-INT confidence scoring | ✅ | ❌ | ❌ |

## Tests

~646 tests, TDD-enforced.

## License

AGPL-3.0
