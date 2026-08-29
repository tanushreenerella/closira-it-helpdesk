# Test Transcript 3 — Escalation Trigger

Scenario: An employee expresses frustration about an outage.

User: The network has been down all morning. This is terrible and completely unacceptable.

IT Support: I'm sorry this has been frustrating. I'm connecting you with Human IT Support right away so they can investigate.  
`[Escalate: true] [Confidence: 1.0] [Escalation reason: Angry sentiment detected]`

Result: PASS — The existing pre-flight sentiment guard escalates before the model call.
