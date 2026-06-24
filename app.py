"""
Streamlit FHIR Patient Q&A Chatbot (POC)

What it does:
1. Fetches 5 patients from the public HAPI FHIR server (GET /Patient)
2. Lets the user pick one patient from a dropdown
3. Fetches that patient's Conditions + MedicationRequests
4. Lets the user ask free-text questions about that patient
5. Sends the question + the patient's flattened clinical data to the
   Claude API, which answers grounded only in that data (simple RAG)

Run locally:
    pip install streamlit requests anthropic
    streamlit run app.py

Requires an Anthropic API key. Set it either as an environment variable
(ANTHROPIC_API_KEY) or, when deployed on Streamlit Community Cloud, in
the app's Secrets as:
    ANTHROPIC_API_KEY = "sk-ant-..."
"""

import streamlit as st
import requests
import os
import anthropic

FHIR_BASE_URL = "https://hapi.fhir.org/baseR4"
CLAUDE_MODEL = "claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# FHIR data fetching
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def fetch_patients(count=5):
    """Fetch N patients from the public FHIR server."""
    url = f"{FHIR_BASE_URL}/Patient"
    params = {"_count": count}

    response = requests.get(url, params=params, timeout=20)
    response.raise_for_status()
    bundle = response.json()

    patients = []
    for entry in bundle.get("entry", []):
        resource = entry.get("resource", {})
        patients.append({
            "id": resource.get("id"),
            "name": extract_name(resource),
            "gender": resource.get("gender", "unknown"),
            "birthDate": resource.get("birthDate", "unknown"),
        })
    return patients


def extract_name(patient_resource):
    names = patient_resource.get("name", [])
    if not names:
        return "Unknown"
    name = names[0]
    given = " ".join(name.get("given", []))
    family = name.get("family", "")
    full = f"{given} {family}".strip()
    return full if full else "Unknown"


@st.cache_data(show_spinner=False)
def fetch_related_resources(resource_type, patient_id):
    """Fetch resources of a given type linked to a patient."""
    url = f"{FHIR_BASE_URL}/{resource_type}"
    params = {"patient": patient_id, "_count": 20}

    response = requests.get(url, params=params, timeout=20)
    if response.status_code != 200:
        return []

    bundle = response.json()
    return [entry["resource"] for entry in bundle.get("entry", [])]


def summarize_condition(resource):
    coding = resource.get("code", {}).get("coding", [{}])
    display = coding[0].get("display") or resource.get("code", {}).get("text", "Unknown condition")
    status = resource.get("clinicalStatus", {}).get("coding", [{}])[0].get("code", "unknown status")
    onset = resource.get("onsetDateTime", "unknown date")
    return f"{display} (status: {status}, onset: {onset})"


def summarize_medication(resource):
    med = resource.get("medicationCodeableConcept", {})
    coding = med.get("coding", [{}])
    display = coding[0].get("display") or med.get("text", "Unknown medication")
    status = resource.get("status", "unknown status")
    authored = resource.get("authoredOn", "unknown date")
    return f"{display} (status: {status}, prescribed: {authored})"


def build_patient_context(patient, conditions, medications):
    """Flatten a patient's data into plain text for the LLM prompt."""
    lines = []
    lines.append(f"Patient: {patient['name']}")
    lines.append(f"Gender: {patient['gender']}")
    lines.append(f"Date of birth: {patient['birthDate']}")

    lines.append("\nConditions:")
    if conditions:
        for c in conditions:
            lines.append(f"- {summarize_condition(c)}")
    else:
        lines.append("- None recorded")

    lines.append("\nMedications:")
    if medications:
        for m in medications:
            lines.append(f"- {summarize_medication(m)}")
    else:
        lines.append("- None recorded")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Claude API call
# ---------------------------------------------------------------------------

def get_claude_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")

    if not api_key:
        # st.secrets raises StreamlitSecretNotFoundError if no secrets.toml
        # exists at all (e.g. local dev with no Secrets configured yet),
        # rather than returning None like a normal dict would.
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY", None)
        except Exception:
            api_key = None

    if not api_key:
        st.error(
            "No Anthropic API key found. Set ANTHROPIC_API_KEY as an environment "
            "variable, or add it to Streamlit Secrets when deployed."
        )
        st.stop()
    return anthropic.Anthropic(api_key=api_key)


def ask_claude(client, patient_context, question):
    system_prompt = (
        "You are a clinical assistant answering questions about a single patient "
        "using only the data provided below. This is synthetic test data, not a "
        "real patient. If the answer isn't in the data, say so clearly instead "
        "of guessing.\n\n"
        f"Patient data:\n{patient_context}"
    )

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=500,
        system=system_prompt,
        messages=[{"role": "user", "content": question}],
    )

    return response.content[0].text


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="FHIR Patient Q&A (POC)", page_icon="🩺")
st.title("🩺 FHIR Patient Q&A Chatbot (POC)")
st.caption(
    "Synthetic patient data from a public FHIR test server. "
    "Not for clinical use — demo purposes only."
)

# --- Load patients ---
with st.spinner("Fetching patients from FHIR server..."):
    try:
        patients = fetch_patients(count=5)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not reach the FHIR server: {e}")
        st.stop()

if not patients:
    st.warning("No patients returned from the server. Try refreshing.")
    st.stop()

# --- Patient selector ---
patient_labels = [f"{p['name']} (ID: {p['id']})" for p in patients]
selected_label = st.selectbox("Select a patient", patient_labels)
selected_patient = patients[patient_labels.index(selected_label)]

st.divider()
st.subheader(f"Selected: {selected_patient['name']}")
st.text(f"Gender: {selected_patient['gender']} | DOB: {selected_patient['birthDate']}")

# --- Fetch this patient's clinical data ---
with st.spinner("Fetching conditions and medications..."):
    conditions = fetch_related_resources("Condition", selected_patient["id"])
    medications = fetch_related_resources("MedicationRequest", selected_patient["id"])

with st.expander("View raw clinical summary used for answers"):
    st.text(build_patient_context(selected_patient, conditions, medications))

st.divider()

# --- Chat ---
st.subheader("Ask a question about this patient")

# Reset chat history if patient changes
if "active_patient_id" not in st.session_state or st.session_state.active_patient_id != selected_patient["id"]:
    st.session_state.active_patient_id = selected_patient["id"]
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.chat_input("e.g. What medications is this patient on?")

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    client = get_claude_client()
    patient_context = build_patient_context(selected_patient, conditions, medications)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_claude(client, patient_context, question)
            st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})