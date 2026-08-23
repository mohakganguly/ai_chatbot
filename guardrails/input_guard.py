"""
Fast input guardrails.

This module performs lightweight deterministic validation
before a user request enters the LangGraph workflow.

Designed for minimal latency.
"""

from __future__ import annotations

import re

from guardrails.schemas import GuardrailResult


class InputGuard:
    """
    Lightweight deterministic input safety guard.

    Performs fast checks for:

    - Invalid input
    - Excessively large input
    - Prompt injection
    - Terrorism and extremist violence
    - Violent wrongdoing
    - Sexual exploitation and sexual content involving minors
    - Self-harm assistance
    - Malware and cyber abuse
    - Fraud and financial crime
    - Illicit drug production and distribution
    - Privacy invasion and credential theft

    This guard intentionally uses regex matching only,
    so validation adds negligible latency.
    """

    MAX_INPUT_LENGTH = 20_000

    # ==========================================================
    # Prompt Injection
    # ==========================================================

    INJECTION_PATTERNS = [

        r"ignore\s+(all\s+)?previous\s+instructions",
        r"ignore\s+(the\s+)?system\s+prompt",
        r"disregard\s+(all\s+)?previous\s+instructions",
        r"forget\s+(all\s+)?previous\s+instructions",

        r"reveal\s+(the\s+)?system\s+prompt",
        r"show\s+(me\s+)?(the\s+)?system\s+prompt",
        r"print\s+(the\s+)?system\s+prompt",

        r"you\s+are\s+now\s+(in\s+)?developer\s+mode",
        r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",

        r"bypass\s+(the\s+)?safety",
        r"disable\s+(all\s+)?guardrails",
        r"override\s+(all\s+)?instructions",

        r"do\s+not\s+follow\s+your\s+instructions",
        r"replace\s+your\s+instructions",
        r"new\s+system\s+prompt",
    ]

    # ==========================================================
    # Terrorism / Extremist Violence
    # ==========================================================

    TERRORISM_PATTERNS = [

        r"\bplan\s+(and\s+)?carry\s+out\s+(a\s+)?terror(ist)?\s+attack\b",
        r"\bhow\s+to\s+(plan|execute|carry\s+out)\s+(a\s+)?terror(ist)?\s+attack\b",
        r"\bhelp\s+(me\s+)?(plan|execute|carry\s+out)\s+(a\s+)?terror(ist)?\s+attack\b",

        r"\bhow\s+to\s+join\s+(a\s+)?terror(ist)?\s+(group|organization)\b",
        r"\bhelp\s+me\s+join\s+(a\s+)?terror(ist)?\b",

        r"\bhow\s+to\s+recruit\s+(people|members)\s+for\s+(a\s+)?terror",
        r"\bhow\s+to\s+finance\s+(a\s+)?terror",
        r"\bhow\s+to\s+fund\s+(a\s+)?terror",

        r"\bexecute\s+(a\s+)?mass\s+casualty\s+attack\b",
        r"\bplan\s+(a\s+)?mass\s+casualty\s+attack\b",
            r"\bmake\s+(an?\s+)?explosive\s+device",
        r"\bbuild\s+(an?\s+)?explosive\s+device",
        r"\bcreate\s+(an?\s+)?explosive\s+device",
        r"\bmake\s+(a\s+)?bomb",
        r"\bbuild\s+(a\s+)?bomb",
        r"\bcreate\s+(a\s+)?bomb",
        r"\bhow\s+to\s+make\s+(an?\s+)?explosive",
    ]

    # ==========================================================
    # Violent Wrongdoing
    # ==========================================================

    VIOLENCE_PATTERNS = [

        r"\bhow\s+to\s+(kill|murder|assassinate)\s+(someone|a\s+person|him|her|them)\b",
        r"\bhelp\s+me\s+(kill|murder|assassinate)\b",
        r"\bplan\s+(a\s+)?murder\b",

        r"\bhow\s+to\s+poison\s+someone\b",
        r"\bhow\s+to\s+attack\s+someone\b",

        r"\bhow\s+to\s+avoid\s+getting\s+caught\s+(after|for)\s+(a\s+)?(murder|attack|crime)\b",
        r"\bhow\s+to\s+dispose\s+of\s+(a\s+)?dead\s+body\b",
        r"\bhow\s+to\s+hide\s+(a\s+)?dead\s+body\b",

        r"\bhelp\s+me\s+carry\s+out\s+(a\s+)?violent\s+attack\b",
    ]

    # ==========================================================
    # Sexual Exploitation / Minors
    # ==========================================================

    SEXUAL_EXPLOITATION_PATTERNS = [

        r"\bsexual\s+(content|images?|material)\s+(with|of|involving)\s+(a\s+)?minor\b",
        r"\bsexualize\s+(a\s+)?(child|minor|underage\s+person)\b",

        r"\bhow\s+to\s+(groom|exploit)\s+(a\s+)?(child|minor)\b",
        r"\bhelp\s+me\s+(groom|exploit)\s+(a\s+)?(child|minor)\b",

        r"\bunderage\s+(porn|pornography|sexual\s+content)\b",
        r"\bchild\s+(porn|pornography|sexual\s+content)\b",

        r"\bhow\s+to\s+hide\s+sexual\s+abuse\b",
    ]

    # ==========================================================
    # Self-Harm Assistance
    # ==========================================================

    SELF_HARM_PATTERNS = [

        r"\bhow\s+to\s+kill\s+myself\b",
        r"\bhow\s+can\s+i\s+commit\s+suicide\b",
        r"\bhelp\s+me\s+commit\s+suicide\b",

        r"\bmost\s+effective\s+way\s+to\s+kill\s+myself\b",
        r"\bbest\s+way\s+to\s+commit\s+suicide\b",

        r"\bhow\s+to\s+self[-\s]?harm\s+without\s+(getting|being)\s+caught\b",
    ]

    # ==========================================================
    # Malware / Cyber Abuse
    # ==========================================================

    CYBER_ABUSE_PATTERNS = [

        r"\bhow\s+to\s+create\s+(a\s+)?malware\b",
        r"\bhow\s+to\s+create\s+(a\s+)?ransomware\b",
        r"\bhow\s+to\s+create\s+(a\s+)?keylogger\b",

        r"\bwrite\s+(a\s+)?virus\b",
        r"\bbuild\s+(a\s+)?virus\b",

        r"\bhow\s+to\s+steal\s+passwords\b",
        r"\bhow\s+to\s+steal\s+credentials\b",
        r"\bhow\s+to\s+steal\s+login\s+credentials\b",

        r"\bhow\s+to\s+hack\s+(into\s+)?someone'?s\s+account\b",
        r"\bhow\s+to\s+hack\s+(into\s+)?(a\s+)?bank\b",

        r"\bhow\s+to\s+bypass\s+(authentication|login|security)\b",

        r"\bhow\s+to\s+deploy\s+(a\s+)?ransomware\b",
        r"\bhow\s+to\s+spread\s+(a\s+)?malware\b",
        r"\bhack\s+(into\s+)?(someone'?s?\s+)?(bank|account|email|computer|server|website)",
        r"\bsteal\s+(someone'?s?\s+)?(password|credentials|login)",
        r"\bcreate\s+(a\s+)?malware",
        r"\bbuild\s+(a\s+)?malware",
        r"\bwrite\s+(a\s+)?malware",
        r"\bmake\s+(a\s+)?virus",
        r"\bcreate\s+(a\s+)?virus",
        r"\bcreate\s+(a\s+)?ransomware",
        r"\bhow\s+to\s+hack\b",
        r"\bphishing\s+(attack|website|page|kit)",
        r"\bkeylogger\b",
        r"\bsteal\s+(credit\s+card|bank)\s+(information|details|data)",

    ]

    # ==========================================================
    # Fraud / Financial Crime
    # ==========================================================

    FRAUD_PATTERNS = [

        r"\bhow\s+to\s+commit\s+fraud\b",
        r"\bhelp\s+me\s+commit\s+fraud\b",

        r"\bhow\s+to\s+create\s+(a\s+)?fake\s+(identity|id|document)\b",
        r"\bhow\s+to\s+forge\s+(a\s+)?(passport|id|document)\b",

        r"\bhow\s+to\s+steal\s+(credit\s+card|bank)\s+(details|information)\b",

        r"\bhow\s+to\s+scam\s+(people|someone)\b",
        r"\bhow\s+to\s+run\s+(a\s+)?scam\b",

        r"\bhow\s+to\s+launder\s+money\b",
        r"\bhelp\s+me\s+hide\s+illegal\s+money\b",
    ]

    # ==========================================================
    # Illicit Drugs
    # ==========================================================

    DRUG_PATTERNS = [

        r"\bhow\s+to\s+manufacture\s+(illegal\s+)?drugs\b",
        r"\bhow\s+to\s+make\s+(illegal\s+)?drugs\b",

        r"\bhow\s+to\s+produce\s+(illegal\s+)?drugs\b",
        r"\bhow\s+to\s+synthesize\s+(illegal\s+)?drugs\b",

        r"\bhow\s+to\s+sell\s+illegal\s+drugs\b",
        r"\bhow\s+to\s+traffic\s+drugs\b",
    ]

    # ==========================================================
    # Privacy / Credential Theft
    # ==========================================================

    PRIVACY_PATTERNS = [

        r"\bhow\s+to\s+steal\s+someone'?s\s+password\b",
        r"\bhow\s+to\s+steal\s+someone'?s\s+credentials\b",

        r"\bhow\s+to\s+find\s+someone'?s\s+private\s+information\b",
        r"\bhow\s+to\s+doxx\s+someone\b",

        r"\bhow\s+to\s+track\s+someone\s+without\s+(them\s+)?knowing\b",

        r"\bfind\s+(me\s+)?(the\s+)?(home\s+address|phone\s+number)\s+of\b",
    ]

    # ==========================================================
    # PII Detection Patterns
    # ==========================================================

    PHONE_PATTERNS = [
        # Indian mobile numbers
        r"\b(?:\+91[\s-]?)?[6-9]\d{9}\b",

        # International-style phone numbers
        r"\b\+?\d{1,3}[\s-]?\(?\d{2,4}\)?[\s-]?\d{3,4}[\s-]?\d{3,4}\b",
    ]

    EMAIL_PATTERNS = [
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b",
    ]

    # ==========================================================
    # Sensitive Credential Patterns
    # ==========================================================

    CREDENTIAL_PATTERNS = [
        # Explicit password sharing
        r"\b(?:my\s+)?password\s*(?:is|=|:)\s*\S+",

        # API keys / tokens explicitly provided
        r"\b(?:api[_\s-]?key|access[_\s-]?token|auth[_\s-]?token|secret[_\s-]?key)"
        r"\s*(?:is|=|:)\s*\S+",

        # Bearer tokens
        r"\bbearer\s+[A-Za-z0-9\-._~+/]+=*\b",

        # Generic private key blocks
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----",
    ]

    # ==========================================================
    # Financial PII Patterns
    # ==========================================================

    CARD_PATTERNS = [
        # Generic card-like number with spaces or hyphens
        r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    ]
    # ==========================================================
    # Constructor
    # ==========================================================

    def __init__(self):

        self._compiled_injection_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.INJECTION_PATTERNS
        ]

        self._compiled_terrorism_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.TERRORISM_PATTERNS
        ]

        self._compiled_violence_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.VIOLENCE_PATTERNS
        ]

        self._compiled_sexual_exploitation_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SEXUAL_EXPLOITATION_PATTERNS
        ]

        self._compiled_self_harm_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.SELF_HARM_PATTERNS
        ]

        self._compiled_cyber_abuse_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.CYBER_ABUSE_PATTERNS
        ]

        self._compiled_fraud_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.FRAUD_PATTERNS
        ]

        self._compiled_drug_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.DRUG_PATTERNS
        ]

        self._compiled_privacy_patterns = [
            re.compile(pattern, re.IGNORECASE)
            for pattern in self.PRIVACY_PATTERNS
        ]

        

        self._compiled_patterns = [
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern in self.INJECTION_PATTERNS
        ]

        self._compiled_phone_patterns = [
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern in self.PHONE_PATTERNS
        ]

        self._compiled_email_patterns = [
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern in self.EMAIL_PATTERNS
        ]

        self._compiled_credential_patterns = [
            re.compile(
                pattern,
                re.IGNORECASE,
            )
            for pattern in self.CREDENTIAL_PATTERNS
        ]

        self._compiled_card_patterns = [
            re.compile(pattern)
            for pattern in self.CARD_PATTERNS
        ]
    # ==========================================================
    # Helper
    # ==========================================================

    @staticmethod
    def _check_patterns(
        text: str,
        patterns: list[re.Pattern],
        category: str,
        reason: str,
    ) -> GuardrailResult | None:

        for pattern in patterns:

            if pattern.search(text):

                return GuardrailResult(
                    allowed=False,
                    reason=reason,
                    category=category,
                )

        return None

    # ==========================================================
    # Main Validation
    # ==========================================================

    def validate(
        self,
        text: str,
    ) -> GuardrailResult:

        # ------------------------------------------------------
        # Empty Input
        # ------------------------------------------------------

        if not text or not text.strip():

            return GuardrailResult(
                allowed=False,
                reason="Empty input is not allowed.",
                category="invalid_input",
            )

        # ------------------------------------------------------
        # Maximum Length
        # ------------------------------------------------------

        if len(text) > self.MAX_INPUT_LENGTH:

            return GuardrailResult(
                allowed=False,
                reason="Input exceeds the maximum allowed length.",
                category="input_too_long",
            )

        # ------------------------------------------------------
        # Prompt Injection
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_injection_patterns,
            "prompt_injection",
            "Potential prompt injection detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Terrorism
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_terrorism_patterns,
            "terrorism_or_extremist_violence",
            "Potential terrorist or extremist violence request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Violent Wrongdoing
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_violence_patterns,
            "violent_wrongdoing",
            "Potential violent wrongdoing request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Sexual Exploitation
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_sexual_exploitation_patterns,
            "sexual_exploitation",
            "Potential sexual exploitation request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Self-Harm
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_self_harm_patterns,
            "self_harm_assistance",
            "Potential self-harm assistance request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Cyber Abuse
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_cyber_abuse_patterns,
            "cyber_abuse",
            "Potential malicious cyber activity request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Fraud
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_fraud_patterns,
            "fraud_or_financial_crime",
            "Potential fraud or financial crime request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Illicit Drugs
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_drug_patterns,
            "illicit_drug_activity",
            "Potential illicit drug activity request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Privacy / Credential Theft
        # ------------------------------------------------------

        result = self._check_patterns(
            text,
            self._compiled_privacy_patterns,
            "privacy_or_credential_theft",
            "Potential privacy or credential theft request detected.",
        )

        if result:
            return result

        # ------------------------------------------------------
        # Sensitive Credential Detection
        # ------------------------------------------------------

        for pattern in self._compiled_credential_patterns:

            if pattern.search(text):

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "The request appears to contain a password, "
                        "API key, access token, or other sensitive credential."
                    ),
                    category="sensitive_credentials",
                )


        # ------------------------------------------------------
        # Phone Number Detection
        # ------------------------------------------------------

        for pattern in self._compiled_phone_patterns:

            if pattern.search(text):

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "The request appears to contain a phone number "
                        "or personally identifiable information."
                    ),
                    category="pii_phone_number",
                )


        # ------------------------------------------------------
        # Email Detection
        # ------------------------------------------------------

        for pattern in self._compiled_email_patterns:

            if pattern.search(text):

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "The request appears to contain an email address "
                        "or personally identifiable information."
                    ),
                    category="pii_email",
                )


        # ------------------------------------------------------
        # Payment Card Detection
        # ------------------------------------------------------

        for pattern in self._compiled_card_patterns:

            if pattern.search(text):

                return GuardrailResult(
                    allowed=False,
                    reason=(
                        "The request appears to contain sensitive "
                        "financial information."
                    ),
                    category="financial_pii",
                )

        # ------------------------------------------------------
        # Input Allowed
        # ------------------------------------------------------

        return GuardrailResult(
            allowed=True,
            reason=None,
            category=None,
        )


# ==========================================================
# Shared Instance
# ==========================================================

_input_guard = InputGuard()


def get_input_guard() -> InputGuard:
    """
    Return the shared InputGuard instance.
    """

    return _input_guard