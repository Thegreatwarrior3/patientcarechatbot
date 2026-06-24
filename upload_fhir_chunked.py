
import os
import json
import time
import requests

# Target FHIR server and headers
TARGET_URL = "h" \
""
HEADERS = {"Content-Type": "application/fhir+json;charset=utf-8"}

# Path to Synthea's FHIR output
output_folder = "./output/fhir"

if not os.path.exists(output_folder):
    print(f"Error: Folder {output_folder} not found.")
    exit(1)

# Byte limit per bundle (2 MB)
CHUNK_BYTE_LIMIT = 2 * 1024 * 1024
# Allowed resource types to include
ALLOWED_TYPES = {"Patient", "Observation", "Condition", "MedicationRequest"}
# Retry settings
MAX_RETRIES = 3
BACKOFF_FACTOR = 1.5


def http_post_json(url, payload, desc="", headers=HEADERS, timeout=300):
    """POST json with retries. Returns response or None."""
    attempt = 0
    while attempt < MAX_RETRIES:
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=timeout)
            return resp
        except requests.exceptions.RequestException as e:
            attempt += 1
            wait = BACKOFF_FACTOR ** attempt
            print(f"Request error ({desc}), attempt {attempt}/{MAX_RETRIES}: {e}. Retrying in {wait:.1f}s")
            time.sleep(wait)
    print(f"Failed to POST {desc} after {MAX_RETRIES} attempts")
    return None


def serialize_bundle(entries):
    bundle = {"resourceType": "Bundle", "type": "transaction", "entry": entries}
    # compact encoding to reduce size
    s = json.dumps(bundle, separators=(",",":"), ensure_ascii=False)
    return s.encode('utf-8')


def send_bundle(entries, desc=""):
    if not entries:
        return True
    data_bytes = serialize_bundle(entries)
    if len(data_bytes) > CHUNK_BYTE_LIMIT:
        print(f"Bundle {desc} unexpectedly exceeds {CHUNK_BYTE_LIMIT} bytes before send: {len(data_bytes)}")
        return False

    resp = http_post_json(TARGET_URL, json.loads(data_bytes.decode('utf-8')), desc=desc)
    if resp is None:
        print(f"✗ {desc} - no response")
        return False
    if resp.status_code in (200, 201):
        print(f"✓ {desc} uploaded (status {resp.status_code})")
        return True
    else:
        print(f"✗ {desc} failed: {resp.status_code}")
        return False


for file_name in os.listdir(output_folder):
    if not file_name.endswith('.json'):
        continue
    if file_name.startswith('hospital') or file_name.startswith('practitioner'):
        continue

    file_path = os.path.join(output_folder, file_name)
    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print(f"Skipping {file_name}: invalid JSON")
            continue

    print(f"Processing {file_name}...")

    # Extract relevant entries
    entries_to_process = []

    if isinstance(data, dict) and data.get('resourceType') == 'Bundle' and 'entry' in data:
        for e in data['entry']:
            resource = e.get('resource')
            if not isinstance(resource, dict):
                continue
            if resource.get('resourceType') in ALLOWED_TYPES:
                # Build minimal entry for transaction: use existing request if present, else construct POST request
                entry = e.copy()
                # Ensure request block exists for transaction: POST to resourceType
                if 'request' not in entry:
                    rtype = resource.get('resourceType')
                    entry['request'] = {
                        'method': 'POST',
                        'url': rtype
                    }
                entries_to_process.append(entry)

    elif isinstance(data, dict) and data.get('resourceType') in ALLOWED_TYPES:
        # single resource file
        entry = {'resource': data, 'request': {'method': 'POST', 'url': data.get('resourceType')}}
        entries_to_process.append(entry)
    else:
        print(f"Skipping {file_name}: no supported resources found")
        continue

    # Accumulate and send bundles that stay under CHUNK_BYTE_LIMIT
    current_entries = []
    for idx, entry in enumerate(entries_to_process):
        # Try adding entry
        current_entries.append(entry)
        size = len(serialize_bundle(current_entries))
        if size >= CHUNK_BYTE_LIMIT:
            # If single entry alone is too big, send it individually to its resource endpoint
            if len(current_entries) == 1:
                resource = current_entries[0].get('resource')
                if resource and isinstance(resource, dict):
                    rtype = resource.get('resourceType')
                    resource_bytes = json.dumps(resource, separators=(",",":"), ensure_ascii=False).encode('utf-8')
                    if len(resource_bytes) > CHUNK_BYTE_LIMIT:
                        print(f"Single resource {rtype} exceeds {CHUNK_BYTE_LIMIT} bytes; uploading as individual POST to /{rtype} and continuing")
                        resp = http_post_json(f"{TARGET_URL}/{rtype}", resource, desc=f"{file_name} - single {rtype}")
                        if resp is None or resp.status_code not in (200, 201):
                            print(f"✗ Failed to upload single large resource {rtype}: {getattr(resp, 'status_code', 'no-response')}")
                        else:
                            print(f"✓ Single large resource {rtype} uploaded")
                        current_entries = []
                        continue
                # If not a huge single resource, still attempt to send (shouldn't normally happen)
                sent = send_bundle(current_entries, desc=f"{file_name} chunk (single entry)")
                current_entries = []
                continue
            # Remove last added entry and send current bundle
            last = current_entries.pop()
            sent = send_bundle(current_entries, desc=f"{file_name} chunk up to idx {idx-1}")
            current_entries = [last]
    # After looping, send any remaining entries
    if current_entries:
        sent = send_bundle(current_entries, desc=f"{file_name} final chunk")

    print(f"Finished processing {file_name}\n")
