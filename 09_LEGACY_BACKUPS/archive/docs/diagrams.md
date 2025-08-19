# QuantoniumOS Architecture & Flow Diagrams

This document provides visual representations of QuantoniumOS's key components and data flows.

## System Architecture

```mermaid
flowchart TD
    subgraph Clients
        Browser[Web Browser]
        Desktop[Desktop App]
        Mobile[Mobile App]
    end

    subgraph ApplicationLayer
        API[Flask API]
        Auth[Auth Service]
    end

    subgraph CoreEngine
        QEngine[Quantum Engine]
        CPP[C++ Core w/ Eigen]
        RustCore[Rust Core]
    end

    subgraph Storage
        Redis[Redis Cache]
        Postgres[PostgreSQL]
        Files[Test Vector Files]
    end

    Clients --> API
    API --> Auth
    API --> QEngine
    API --> CPP
    API --> RustCore
    QEngine --> CPP
    QEngine --> Redis
    CPP --> Redis
    CPP --> Postgres
    RustCore --> Postgres
    Auth --> Postgres
```

## Encryption Pipeline

```mermaid
flowchart LR
    Input[Input Message] --> KeyDerivation[Key Derivation]
    UserKey[User Key] --> KeyDerivation
    KeyDerivation --> KeySchedule[Key Schedule]
    KeySchedule --> WaveGeneration[Wave Pattern Generation]
    WaveGeneration --> PhaseTransformation[Phase Transformation]
    PhaseTransformation --> AmplitudeModulation[Amplitude Modulation]
    AmplitudeModulation --> XOROperation[XOR with Message]
    XOROperation --> BitRotation[Bit Rotation]
    BitRotation --> Signature[Add Signature]
    Signature --> TokenAddition[Add Token]
    TokenAddition --> FinalCiphertext[Final Ciphertext]
```

## Cross-Implementation Validation Flow

```mermaid
flowchart TD
    Generator[Test Vector Generator] --> TestVectors[(Test Vectors)]
    TestVectors --> CPPImpl[C++ Implementation]
    TestVectors --> PyImpl[Python Implementation]
    TestVectors --> RustImpl[Rust Implementation]
    CPPImpl --> CPPResult[C++ Result]
    PyImpl --> PyResult[Python Result]
    RustImpl --> RustResult[Rust Result]
    CPPResult --> Comparator{Byte-by-Byte Comparison}
    PyResult --> Comparator
    RustResult --> Comparator
    Comparator --> Match[Match ]
    Comparator --> Mismatch[Mismatch ]
    Mismatch --> Debug[Debug Diagnostics]
```

## Resonance Encryption Class Structure

```mermaid
classDiagram
    class ResonanceEncryptor {
        -string key
        -bytes keyHash
        -int iterations
        +constructor(string key)
        +encrypt(bytes message) bytes
        +decrypt(bytes ciphertext) bytes
        -generateKeystream(bytes seed, int length) bytes
        -validateSignature(bytes ciphertext) bool
    }

    class GeometricWaveHash {
        +hash(bytes data) bytes
        -transformData(bytes chunk) bytes
        -finalizeHash(array chunks) bytes
    }

    class StatisticalValidator {
        +validateEntropy(bytes data) bool
        +runNIST_SP800_22(bytes data) Result
        +checkAvalancheEffect(bytes data1, bytes data2) float
    }

    ResonanceEncryptor --> GeometricWaveHash: uses
    ResonanceEncryptor --> StatisticalValidator: validates with
```

## CI/CD Pipeline

```mermaid
flowchart TD
    Push[Git Push] --> MainCI{Green Wall CI}
    MainCI --> PythonTests[Python Core Validation]
    MainCI --> CPPBuildLinux[C++ Build (Linux)]
    MainCI --> CPPBuildWin[C++ Build (Windows)]
    MainCI --> IntegrationTests[Integration Tests]
    MainCI --> GenerateArtifacts[Generate Artifacts]

    GenerateArtifacts --> TestVectors[(Test Vectors)]
    GenerateArtifacts --> Benchmarks[(Benchmarks)]

    TestVectors --> CrossValidation{Cross-Validation CI}
    CrossValidation --> ValStatus[Validation Status]

    PythonTests --> GreenWallStatus{Green Wall Status}
    CPPBuildLinux --> GreenWallStatus
    CPPBuildWin --> GreenWallStatus
    IntegrationTests --> GreenWallStatus
    ValStatus --> GreenWallStatus

    GreenWallStatus --> Success[Success ]
    GreenWallStatus --> Failure[Failure ]
```
