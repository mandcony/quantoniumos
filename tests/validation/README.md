# Validation Tests Notes

## Crypto AEAD streaming tests

- The large-payload AEAD test is marked `slow` and defaults to 1 MiB.
- For quick local runs, either deselect slow tests or reduce payload size with an env var.

Examples (PowerShell):

- Run all tests except slow:

```
pytest -m "not slow"
```

- Keep the slow test but reduce its size to 128 KiB:

```
$Env:RFT_CRYPTO_TEST_PAYLOAD = 131072
pytest tests/validation/test_enhanced_rft_crypto_streaming.py::test_aead_roundtrip_megabyte_payload
```

The regular fast test `test_aead_roundtrip_fast_payload_64kb` always runs and keeps CI/local cycles snappy.
