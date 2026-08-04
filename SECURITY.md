# Security Policy

## Purpose

This repository is a defensive cybersecurity research prototype for network anomaly detection. It is not a production security product.

## Reporting concerns

If you find a vulnerability, unsafe behavior, data exposure risk, or documentation that could enable misuse, please open a GitHub issue with a clear description.

For sensitive reports, avoid posting secrets, private telemetry, credentials, or exploit instructions publicly.

## Responsible-use boundaries

This project should be used for:

- Defensive cybersecurity research.
- Academic experimentation.
- Educational demonstrations.
- Reproducible model evaluation.

This project should not be used for:

- Evading monitoring systems.
- Hiding malicious activity.
- Processing private logs without authorization.
- Making unsupported claims about production security performance.

## Data handling

The default workflow uses synthetic telemetry. If you adapt the project to real data:

- Remove private identifiers.
- Follow institutional and legal data policies.
- Document dataset permissions.
- Avoid committing raw sensitive logs.
- Keep secrets out of the repository.

## Production caution

Production deployment requires additional engineering, monitoring, privacy review, incident-response integration, and human oversight.
