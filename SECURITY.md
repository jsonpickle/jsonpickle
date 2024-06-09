# Security Policy

## Security contact information

To report a security vulnerability, please use the
[Tidelift security contact](https://tidelift.com/security).
Tidelift will coordinate the fix and disclosure.

## Reporting a vulnerability

Even when unsure whether the bug in question is an exploitable
vulnerability, it is recommended to send the report and not to
discuss the issue anywhere else.

Vulnerabilities are expected to be discussed _only_ via email,
and not in public, until an official release to address the
vulnerability is available.

Examples for details to include:

- Ideally a short description (or a script) to demonstrate an
  exploit.
- The affected platforms and scenarios.
- The name and affiliation of the security researchers who are
  involved in the discovery, if any.
- Whether the vulnerability has already been disclosed.
- How long an embargo would be required to be safe.

## Supported Versions

There are no official "Long Term Support" versions in jsonpickle.
Instead, the maintenance track (i.e. the versions based on the
most recently published feature release, also known as ".0"
version) sees occasional updates with bug fixes.

Fixes to vulnerabilities are made for the maintenance track for
the latest feature release. The jsonpickle project makes no formal
guarantee for any older maintenance tracks to receive updates.
In practice, though, critical vulnerability fixes can be applied not
only to the most recent track, but to at least a couple more
maintenance tracks if requested by users.

## Security

The jsonpickle module **is not secure**.  Only unpickle data you trust.

It is possible to construct malicious pickle data which will **execute
arbitrary code during unpickling**.  Never unpickle data that could have come
from an untrusted source, or that could have been tampered with.

Consider signing data with an HMAC if you need to ensure that it has not
been tampered with.

Safer deserialization approaches, such as reading JSON directly,
may be more appropriate if you are processing untrusted data.
