# Test Transcript 5 — IT Support Session Summary

Scenario: A completed support session produces a structured summary.

```json
{
  "session_id": "a3f1b2c4-...",
  "employee_intent": "Employee cannot send email from a Windows 11 laptop and needs to send invoices.",
  "key_details_collected": {
    "employee": "Priya, Finance",
    "issue": "Outlook cannot connect; invoice delivery is blocked",
    "device_system": "Windows 11 laptop; Outlook connection error"
  },
  "sop_gaps": [],
  "recommended_next_action": "Human IT Support should investigate the Outlook connection issue.",
  "escalated": false,
  "escalation_reason": null
}
```

Result: PASS — The summary captures employee, issue, device or system, SOP gaps, and the recommended action.
