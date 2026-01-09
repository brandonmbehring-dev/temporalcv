# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take security vulnerabilities seriously. If you discover a security issue,
please report it through GitHub's private vulnerability reporting feature:

1. Go to the [Security Advisories](https://github.com/bbehring/temporalcv/security/advisories) page
2. Click "Report a vulnerability"
3. Fill out the form with details about the vulnerability

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Any suggested fixes (optional)

### Response Timeline

- **Initial response**: Within 48 hours
- **Status update**: Within 5 business days
- **Resolution target**: Within 30 days for critical issues

### What to Expect

1. You will receive an acknowledgment within 48 hours
2. We will investigate and provide a status update
3. Once fixed, we will coordinate disclosure timing with you
4. Credit will be given in the release notes (unless you prefer anonymity)

## Security Considerations

### Data Handling

temporalcv processes numerical arrays for cross-validation and statistical testing.
The library:

- Does NOT store or transmit user data
- Does NOT make network requests
- Does NOT execute arbitrary code from inputs
- Operates entirely in-memory

### Dependencies

We maintain upper bounds on dependencies to prevent unexpected breaking changes.
Security updates to dependencies are incorporated in patch releases.

### Known Limitations

- Statistical tests assume valid numerical input; validation is the user's responsibility
- Large arrays may cause memory issues on constrained systems
- No sandboxing of user-provided model objects

## Best Practices for Users

1. Validate input data before passing to temporalcv functions
2. Use virtual environments to isolate dependencies
3. Keep temporalcv and its dependencies updated
4. Review model objects before use (if from external sources)
