from __future__ import annotations

import re
from typing import Dict


EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"\+?\d[\d\-()\s]{7,}\d")


def extract_resume_metadata(text: str) -> Dict[str, str]:
    email = EMAIL_RE.search(text)
    phone = PHONE_RE.search(text)
    return {
        "candidate_name": "",
        "email": email.group(0) if email else "",
        "phone": phone.group(0) if phone else "",
    }


def extract_jd_metadata(text: str) -> Dict[str, str]:
    return {"job_title": "", "company": "", "source": ""}


