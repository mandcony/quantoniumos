# Slow Tests

This folder holds long-running validations (e.g., NIST STS). Mark tests with:

```python
import pytest
@pytest.mark.slow
```

Wire a separate CI job or manual workflow to run them and upload XML reports.