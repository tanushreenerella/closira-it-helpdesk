# Closira IT Helpdesk AI Agent

## System prompt

The assistant is a concise, professional IT support analyst for employees. It answers only from `backend/sop.json`, which covers accounts and passwords, Wi-Fi and network connectivity, VPN, managed devices, email, and approved software or access.

It must never request passwords, one-time codes, recovery codes, or sensitive email content. Suspected phishing, malware, account compromise, lost or stolen devices, high-impact outages, privileged access requests, and unsupported questions are escalated to Human IT Support.

The FAQ reply uses this JSON contract (Groq writes text only; routing fields are ignored):

```json
{
  "response": "<support response>",
  "sop_gap": false
}
```

## Qualification (slot filling)

After a supported, non-greeting request the assistant first asks for an optional `requester`
(name or employee ID — "skip" declines it; stored on the ticket), then fills `issue_category`,
`problem_details`, `error_message`, `device_os`, `impact`. Groq extracts any slots stated in
each message (extraction only); the assistant then asks only for the missing ones, one at a
time, phrased for the issue category. A vague description ("not working properly") gets one
category-specific follow-up. Once all slots are filled, the matching SOP `self_help_steps`
are offered and the employee is asked whether they worked. A ticket is raised only if the
problem is unresolved or the employee asks for one; its severity comes from the session's
peak risk and the reported impact (`router.ticket_severity`).

No secrets are requested during qualification.

## Grounding and escalation

The assistant marks questions outside the SOP as `sop_gap`. The existing unanswered-question counter and escalation routing remain unchanged. The existing sentiment and explicit-human-request checks also remain unchanged.

## Turn pipeline

```text
run_turn(state, msg)
  classify  -> ML intent + emotion -> risk          (backend/ml, router.risk_score)
  route     -> faq | qualify | escalate | summary   (router.route; keyword rules are hard overrides)
  handler   -> faq_handler (Groq writes the reply) | qualify_handler | escalate_handler | summary_handler
```

Groq only generates FAQ reply text (`response`, `sop_gap`), extracts slot values, and writes the session summary. It never
chooses a route; an `sop_gap` it reports is passed to the router as a rule flag.
