# Test Transcript 4 — IT Issue Qualification

Scenario: The AI collects three support details one at a time.

IT Support: What is your name or employee identifier, if you are comfortable sharing it?

User: Priya, Finance.

IT Support: Please describe the issue and the impact it is having on your work.

User: I cannot send email, so I cannot send today’s invoices.

IT Support: What device and operating system are you using, and what exact error message do you see?

User: Windows 11 laptop. Outlook says it cannot connect to the server.

IT Support: Thank you. I have your employee information, issue details, and device details. I'll prepare a short session summary for Human IT Support.  
`[Stage: summary] [stage_complete: true]`

Result: PASS — The existing qualification stage retains all three answers and then routes to the summary stage.
