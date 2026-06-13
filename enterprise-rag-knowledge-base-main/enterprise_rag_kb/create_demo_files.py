import os
import json
import csv
import subprocess
import sys

def install_fpdf():
    try:
        import fpdf
    except ImportError:
        print("Installing fpdf2 for PDF generation...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2"])
        print("fpdf2 installed.")

install_fpdf()
from fpdf import FPDF

os.makedirs('demo_files', exist_ok=True)

# 1. TXT File
txt_content = """AlphaCorp Enterprise - Employee Handbook
Effective Date: January 1, 2026

1. Remote Work Policy:
Employees are allowed to work remotely up to 3 days a week. Core hours are 10:00 AM to 3:00 PM EST.

2. Leave Policy:
All full-time employees are entitled to 20 days of paid time off (PTO) and 10 sick days per year.

3. Code of Conduct:
Respect, integrity, and innovation are the core values of AlphaCorp. Harassment of any kind will result in immediate termination.
"""
with open('demo_files/employee_handbook.txt', 'w', encoding='utf-8') as f:
    f.write(txt_content)

# 2. Markdown File
md_content = """# Q3 Financial Report 2025

## Executive Summary
AlphaCorp saw a significant increase in revenue during the third quarter of 2025, driven mainly by the launch of our new AI-powered analytics platform.

## Key Metrics
* **Total Revenue:** $45.2 Million (Up 12% YoY)
* **Net Income:** $12.4 Million
* **Operating Margin:** 24%
* **New Customer Acquisitions:** 1,250

## Future Outlook
We expect Q4 to maintain this momentum, projecting an additional 8% growth in recurring revenue.
"""
with open('demo_files/financial_report.md', 'w', encoding='utf-8') as f:
    f.write(md_content)

# 3. CSV File
csv_content = [
    ["server_id", "status", "region", "cpu_usage", "memory_usage"],
    ["srv-us-east-1a", "active", "US-East", "45%", "60%"],
    ["srv-us-east-1b", "maintenance", "US-East", "0%", "0%"],
    ["srv-eu-west-2a", "active", "EU-West", "78%", "82%"],
    ["srv-ap-south-1a", "active", "AP-South", "32%", "41%"]
]
with open('demo_files/server_status.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(csv_content)

# 4. JSON File
json_content = {
    "project_name": "Project Titan",
    "status": "In Progress",
    "deadline": "2026-10-15",
    "team_members": [
        {"name": "Sarah Connor", "role": "Project Manager", "email": "sarah.c@alphacorp.com"},
        {"name": "John Smith", "role": "Lead Developer", "email": "john.s@alphacorp.com"}
    ],
    "milestones": [
        {"id": 1, "name": "Architecture Design", "completed": True},
        {"id": 2, "name": "Beta Release", "completed": False}
    ]
}
with open('demo_files/project_titan_info.json', 'w', encoding='utf-8') as f:
    json.dump(json_content, f, indent=4)

# 5. PDF File
class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 15)
        self.cell(0, 10, "Security Architecture Review", border=False, ln=True, align="C")
        self.ln(10)

pdf = PDF()
pdf.add_page()
pdf.set_font("helvetica", size=12)

pdf_text = """Security Architecture Review - 2026

1. Authentication Mechanisms:
The system employs OAuth 2.0 and OpenID Connect for secure user authentication. Multi-factor authentication (MFA) is mandatory for all administrative access.

2. Data Encryption:
All sensitive data is encrypted at rest using AES-256 bit encryption. Data in transit is secured using TLS 1.3 to prevent interception.

3. Threat Detection:
An Intrusion Detection System (IDS) is actively monitoring network traffic. Any anomalies are immediately reported to the Security Operations Center (SOC).

4. Incident Response Plan:
In the event of a breach, the system goes into lockdown mode, disabling all external API endpoints until verified by the security team.
"""
pdf.multi_cell(0, 10, txt=pdf_text)
pdf.output("demo_files/security_architecture.pdf")

print("All 5 demo files (TXT, MD, CSV, JSON, PDF) created successfully in the 'demo_files' directory!")
