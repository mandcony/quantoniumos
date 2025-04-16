# 🧠 Quantonium OS – Cloud Runtime API

*Quantonium OS Cloud Runtime* is a secure API gateway to the Quantonium symbolic computing platform. This API provides access to quantum-inspired encryption, resonance-based calculations, and secure container operations through authenticated endpoints.

## 🔐 Authentication

All API endpoints require token-based authentication using the `X-API-Key` header:

```
X-API-Key: your_api_key_here
```

The API key should be stored in the `QUANTONIUM_API_KEY` environment variable.

## 📦 API Endpoints

### Status Check
```
GET /
```

Returns the current status of the Quantonium OS Cloud Runtime.

**Response:**
```json
{
  "name": "Quantonium OS Cloud Runtime",
  "status": "operational",
  "version": "1.0.0"
}
```

### Symbolic Encryption
```
POST /encrypt
```

Encrypt plaintext using symbolic waveform-based XOR encryption.

**Request:**
```json
{
  "plaintext": "hello world",
  "key": "symbolic-key"
}
```

**Response:**
```json
{
  "ciphertext": "MgDA7hq6XlUjIEY=",
  "timestamp": 1744809076,
  "sha256": "0757509ce39adda09ad0bc80ece82e81923dade35d8331af857e7dd425137e5b"
}
```

### Resonance Fourier Transform
```
POST /simulate/rft
```

Perform a Resonance Fourier Transform (RFT) on an input waveform.

**Request:**
```json
{
  "waveform": [0.1, 0.5, 0.9, 0.5, 0.1, 0.5, 0.9, 0.5]
}
```

**Response:**
```json
{
  "frequencies": {
    "freq_0": 0.5,
    "freq_2": 0.2,
    "freq_6": 0.2
  },
  "timestamp": 1744809081,
  "sha256": "be633ade79421b43d86e9d0c1e6fad56f1e968eb00f4b93422ad35b7daac2bfc"
}
```

### Quantum Entropy Generation
```
POST /entropy/sample
```

Generate quantum-inspired entropy for cryptographic operations.

**Request:**
```json
{
  "amount": 64
}
```

**Response:**
```json
{
  "entropy": "y85WF5PcX/fwa2MwZFuUEL8T8OU/KMkShiKRGnCVT4mtEJp2J2JVq9yGc6TZ63U/gad+UVr5mc1lPAJ/tlcUqA==",
  "timestamp": 1744809085,
  "sha256": "03a8378ca4b81401d87378785abe487dfd65f03cf83534e8821c5ac87ab26ec7"
}
```

### Symbolic Container Unlocking
```
POST /container/unlock
```

Unlock a symbolic container using a waveform key and verification hash.

**Request:**
```json
{
  "waveform": [0.2, 0.7, 0.3],
  "hash": "d6a88f4f..."
}
```

**Response:**
```json
{
  "unlocked": false,
  "timestamp": 1744809089,
  "sha256": "9717c88ed04cf2cf96525f296374ae42e9560908ea38338cb669ebfd54fb708d"
}
```

## 🛠️ Running the Server

Start the server with `gunicorn`:

```bash
gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app
```

Or use the provided shell script:

```bash
./start.sh
```

## 🔒 Security 

All API responses include:
1. A Unix timestamp (`timestamp`)
2. A SHA-256 hash signature (`sha256`)

This allows verification of response integrity and prevention of replay attacks.

## ⚡ Symbolic Stack Layers

| Layer       | Purpose                         | Endpoint           |
|-------------|----------------------------------|-------------------|
| Encryption  | Waveform-driven XOR              | `/encrypt`        |
| Analysis    | Resonance Fourier Transform      | `/simulate/rft`   |
| Randomness  | QRNG-style entropy for keys      | `/entropy/sample` |
| Containers  | Secure container access control  | `/container/unlock` |