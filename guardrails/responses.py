"""
Category-specific responses for blocked requests.
"""

GUARDRAIL_RESPONSES = {

    "prompt_injection": """
I can't follow requests intended to override system instructions,
disable safety mechanisms, or reveal protected system information.

You can still ask for help with legitimate tasks or questions.
""",

    "terrorism_or_extremist_violence": """
I can't help with planning, carrying out, or facilitating terrorist
or extremist violence.

I can help with terrorism prevention, public safety, emergency
preparedness, or the historical and social impact of terrorism.
""",

    "cybercrime": """
I can't help with unauthorized access to accounts, systems, or networks.

I can help with ethical hacking in a legal environment,
cybersecurity fundamentals, CTF challenges, or securing your own systems.
""",

    "malware": """
I can't provide instructions for creating, deploying, or distributing malware.

I can help explain malware from a defensive perspective, including
malware detection, analysis, and protecting systems.
""",

    "explosives_or_weapons": """
I can't provide instructions for constructing dangerous explosives
or weapons intended to harm people.

I can help with safety information, legal regulations, or general
scientific concepts at a non-actionable level.
""",

    "self_harm": """
I'm sorry you're dealing with something that led to this question.

I can't provide instructions that could help you harm yourself.
Please consider reaching out to someone you trust or local emergency
services if you're in immediate danger.
""",

    "sexual_exploitation": """
I can't help with requests involving sexual exploitation or abuse.

I can provide information about consent, online safety, legal protections,
or reporting harmful content.
""",
}


def get_guardrail_response(category: str) -> str:
    """
    Return a safe, category-specific response.
    """

    return GUARDRAIL_RESPONSES.get(
        category,
        """
I can't assist with this particular request.

However, I may be able to help with a safe and legitimate alternative.
""",
    )