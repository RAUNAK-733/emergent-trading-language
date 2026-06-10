# Security Policy

## Supported Version

Security fixes are applied to the latest version on the `master` branch.

## Reporting a Vulnerability

Please report suspected vulnerabilities privately through GitHub's
**Security > Report a vulnerability** feature when available. Do not open a
public issue for credentials, unsafe deserialization concerns, dependency
vulnerabilities, or other sensitive findings.

Include:

- affected files or versions;
- reproduction steps;
- expected impact;
- any suggested mitigation.

This project loads local PyTorch checkpoints. Only load checkpoints created by
trusted project runs; serialized model files from unknown sources may be unsafe.
