"""
Streamlit FHIR Patient Q&A Chatbot (POC)

What it does:
1. Fetches 5 patients from the public HAPI FHIR server (GET /Patient)
2. Lets the user pick one patient from a dropdown
3. Fetches that patient's Conditions + MedicationRequests
4. Lets the user ask free-text questions about that patient
5. Sends the question + the patient's flattened clinical data to the
   Claude API, which answers grounded only in that data (simple RAG)
6. If the user types "send summary to <email>", emails the conversation
   summary to that address via Gmail SMTP

Run locally:
    pip install streamlit requests anthropic
    streamlit run app.py

Requires secrets (env vars locally, Streamlit Secrets when deployed):
    ANTHROPIC_API_KEY   — Anthropic API key
    EMAIL_SENDER        — your Gmail address
    EMAIL_PASSWORD      — Gmail App Password (not your Gmail login password)
"""

import streamlit as st
import requests
import os
import re
import anthropic
from email_sender import send_summary_email

FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"
CLAUDE_MODEL  = "claude-sonnet-4-6"

# Regex to detect "send summary to someone@example.com" anywhere in a message
EMAIL_SEND_PATTERN = re.compile(
    r"send\s+(?:summary|report|this)?\s*to\s+([\w.+-]+@[\w.-]+\.[a-z]{2,})",
    re.IGNORECASE
)


# ---------------------------------------------------------------------------
# FHIR helpers
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_patients(count=5):
    url    = f"{FHIR_BASE_URL}/Patient"
    params = {"_count": count}
    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    bundle = response.json()
    patients = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        patients.append({
            "id":        resource.get("id"),
            "name":      extract_name(resource),
            "gender":    resource.get("gender", "unknown"),
            "birthDate": resource.get("birthDate", "unknown"),
        })
    return patients


def extract_name(resource):
    names = resource.get("name", [])
    if not names:
        return "Unknown"
    n = names[0]
    given  = " ".join(n.get("given", []))
    family = n.get("family", "")
    full   = f"{given} {family}".strip()
    return full or "Unknown"


@st.cache_data(show_spinner=False)
def fetch_related_resources(resource_type, patient_id):
    url    = f"{FHIR_BASE_URL}/{resource_type}"
    params = {"patient": patient_id, "_count": 20}
    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        return []
    return [e["resource"] for e in response.json().get("entry", [])]


def summarize_condition(r):
    coding  = r.get("code", {}).get("coding", [{}])
    display = coding[0].get("display") or r.get("code", {}).get("text", "Unknown condition")
    status  = r.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "unknown")
    onset   = r.get("onsetDateTime", "unknown date")
    return f"{display} (status: {status}, onset: {onset})"


def summarize_medication(r):
    med     = r.get("medicationCodeableConcept", {})
    coding  = med.get("coding", [{}])
    display = coding[0].get("display") or med.get("text", "Unknown medication")
    status  = r.get("status", "unknown")
    authored = r.get("authoredOn", "unknown date")
    return f"{display} (status: {status}, prescribed: {authored})"


def build_patient_context(patient, conditions, medications):
    lines = [
        f"Patient: {patient['name']}",
        f"Gender: {patient['gender']}",
        f"Date of birth: {patient['birthDate']}",
        "\nConditions:",
    ]
    lines += [f"- {summarize_condition(c)}" for c in conditions] or ["- None recorded"]
    lines.append("\nMedications:")
    lines += [f"- {summarize_medication(m)}" for m in medications] or ["- None recorded"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude helpers
# ---------------------------------------------------------------------------

def get_claude_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        except Exception:
            api_key = None
    if not api_key:
        st.error("No Anthropic API key found. Set ANTHROPIC_API_KEY in secrets.")
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def ask_claude(client, patient_context, question):
    system_prompt = (
        "You are a clinical assistant answering questions about a single patient "
        "using only the data provided below. This is synthetic test data, not a "
        "real patient. If the answer isn't in the data, say so clearly.\n\n"
        f"Patient data:\n{patient_context}"
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )
    return response.content[0].text


def generate_summary(client, patient_name, messages):
    """Ask Claude to produce a concise summary of the conversation so far."""
    conversation_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in messages
    )
    prompt = (
        f"Summarize the following clinical Q&A conversation about patient {patient_name} "
        "into a concise, professional email-ready summary. Include key findings, "
        "medications discussed, and any notable clinical points raised.\n\n"
        f"Conversation:\n{conversation_text}"
    )
    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


# ---------------------------------------------------------------------------
# Email intent detection
# ---------------------------------------------------------------------------

def detect_email_intent(user_message):
    """
    Returns an email address string if the user wants to send a summary,
    otherwise returns None.

    Matches phrases like:
      "send summary to vivek@gmail.com"
      "email this to doctor@hospital.org"
      "send to me@example.com"
    """
    match = EMAIL_SEND_PATTERN.search(user_message)
    return match.group(1) if match else None


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="FHIR Patient Q&A", page_icon="🩺")
st.title("🩺 FHIR Patient Q&A Chatbot")
st.caption("Synthetic patient data — not for clinical use.")

# Load patients
with st.spinner("Fetching patients from FHIR server..."):
    try:
        patients = fetch_patients(count=5)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the FHIR server: {e}")
        st.stop()

if not patients:
    st.warning("No patients returned. Try refreshing.")
    st.stop()

# Patient selector
patient_labels  = [f"{p['name']} (ID: {p['id']})" for p in patients]
selected_label  = st.selectbox("Select a patient", patient_labels)
selected_patient = patients[patient_labels.index(selected_label)]

st.divider()
st.subheader(f"Selected: {selected_patient['name']}")
st.text(f"Gender: {selected_patient['gender']} | DOB: {selected_patient['birthDate']}")

# Fetch clinical data
with st.spinner("Fetching conditions and medications..."):
    conditions  = fetch_related_resources("Condition",         selected_patient["id"])
    medications = fetch_related_resources("MedicationRequest", selected_patient["id"])

with st.expander("View raw clinical data used for answers"):
    st.text(build_patient_context(selected_patient, conditions, medications))

st.divider()
st.subheader("Ask a question about this patient")
st.caption("Tip: type **'send summary to you@email.com'** to email the conversation summary.")

# Reset chat if patient changes
if ("active_patient_id" not in st.session_state
        or st.session_state.active_patient_id != selected_patient["id"]):
    st.session_state.active_patient_id = selected_patient["id"]
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask a question, or type 'send summary to you@email.com'")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    client = get_claude_client()

    # --- Email intent branch ---
    recipient = detect_email_intent(user_input)
    if recipient:
        if len(st.session_state.messages) < 3:
            reply = "There isn't enough conversation yet to summarize. Ask a few questions first, then try sending again."
        else:
            with st.chat_message("assistant"):
                with st.spinner(f"Generating summary and sending to {recipient}..."):
                    summary = generate_summary(
                        client,
                        selected_patient["name"],
                        # exclude the send-request message itself from the summary
                        st.session_state.messages[:-1],
                    )
                    success, detail = send_summary_email(
                        patient_name=selected_patient["name"],
                        summary_text=summary,
                        recipient_override=recipient,
                    )
                    if success:
                        reply = f"Summary sent to **{recipient}**.\n\n**Summary:**\n\n{summary}"
                    else:
                        reply = f"Could not send the email: {detail}"
                st.markdown(reply)

    # --- Normal Q&A branch ---
    else:
        patient_context = build_patient_context(selected_patient, conditions, medications)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                reply = ask_claude(client, patient_context, user_input)
            st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})