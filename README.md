# AI Paper Renamer

AI Paper Renamer automatically renames academic PDF files using AI-extracted titles from the first page.

On **macOS**, you can select one or more PDFs in Finder and rename them with a single keyboard shortcut via Automator Quick Action.

The Python script itself is cross-platform and works on **Linux** (and Windows) as a standalone command-line tool.

All required code is embedded below—just copy and paste.

---

## 1. Prerequisites

### 1.1 Install Python 3

Check Python 3 in Terminal:

```sh
python3 --version
```

If missing, install:

```sh
brew install python
```

### 1.2 Install dependencies

```sh
python3 -m pip install --upgrade pip
python3 -m pip install pymupdf requests
```

---

## 2. Create a script folder

On macOS, use `~/Library/Application Support/`:

```sh
mkdir -p "$HOME/Library/Application Support/AI Paper Renamer"
```

On Linux, you can use any directory you prefer (e.g., `~/.local/share/ai-paper-renamer`).

---

## 3. Create the Python renaming script

Create `rename_script.py` in the folder from step 2 (on macOS: `~/Library/Application Support/AI Paper Renamer/rename_script.py`).

The script is cross-platform and works on macOS, Linux, and Windows.

Choose **one** of the two versions below, and **replace `API_KEY` with your own key** in the configuration section:

### Option A: Google Gemini API (direct)

```python
#!/usr/bin/env python3
import fitz  # PyMuPDF
import requests
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import re

# --- Configuration ---
# Replace with your Gemini API Key
API_KEY = "YOUR_GEMINI_API_KEY"
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
BASE_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

# Thread-safe lock to prevent jumbled output from multiple threads
print_lock = threading.Lock()

def safe_print(message):
    """Thread-safe print"""
    with print_lock:
        print(message)

def fetch_with_retry(url, options, max_retries=5):
    """API call with exponential backoff retry"""
    for i in range(max_retries):
        try:
            response = requests.request(**options)
            response.raise_for_status()  # Raise exception for 4xx/5xx
            return response
        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                delay = (2 ** i) * 1 + (time.time() % 1)  # Exponential backoff + jitter
                safe_print(f"[Warning] API call failed ({e}). Retry {i + 1}/{max_retries}, wait {delay:.2f}s...")
                time.sleep(delay)
                continue
            raise

def extract_title_from_pdf(pdf_path):
    """
    Load the first page of the PDF, dynamically crop the top area, and call VLM to extract the title.
    Returns: The extracted and normalized title string, or None if failed.
    """
    base_filename = os.path.basename(pdf_path)
    safe_print(f"[{threading.current_thread().name}] Processing: {base_filename}")

    if API_KEY == "YOUR_GEMINI_API_KEY":
        safe_print(f"[{threading.current_thread().name}] [Error] Please set your API_KEY in the script.")
        return None

    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]

        safe_print(f"[{threading.current_thread().name}] 1. Loading PDF and calculating crop area...")
        scale = 3.0
        matrix = fitz.Matrix(scale, scale)

        # --- Dynamic cropping logic ---
        text_blocks = page.get_text("dict")["blocks"]
        top_content_y_pdf_unit = float("inf")

        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if line["spans"]:
                        top_content_y_pdf_unit = min(top_content_y_pdf_unit, line["bbox"][1])

        if top_content_y_pdf_unit == float("inf"):
            safe_print(f"[{threading.current_thread().name}] [Warning] No text detected, using default crop.")
            top_content_y_pdf_unit = 0

        pdf_unit_buffer = 5
        start_y_pdf_unit = max(0, top_content_y_pdf_unit - pdf_unit_buffer)

        max_content_height_ratio = 0.40
        max_crop_height = int(page.rect.height * scale * max_content_height_ratio)

        crop_rect_pdf = fitz.Rect(
            page.rect.x0,
            start_y_pdf_unit,
            page.rect.x1,
            min(page.rect.y1, start_y_pdf_unit + (max_crop_height / scale))
        )

        pix = page.get_pixmap(matrix=matrix, clip=crop_rect_pdf)
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        safe_print(f"[{threading.current_thread().name}] Crop done. Image size: {pix.width}x{pix.height}.")

        # --- VLM API call ---
        safe_print(f"[{threading.current_thread().name}] 2. Calling Gemini VLM API to extract title...")

        system_prompt_text = (
            "You are a professional academic paper title extraction assistant. Only identify the title of the academic paper from the provided image."
            "Return the extracted title as plain text, **strictly ensuring the result is on a single line with no newline characters**."
            "**For words that appear in small caps or all caps in the original paper, normalize them to standard case (Title Case),**"
            "**unless the word is a clear and recognized all-caps acronym (e.g., 'NLP', 'AI', 'LLM', 'DNN', etc.),**"
            "in which case, preserve its uppercase form."
        )
        user_query_text = "Please extract the exact title from the first page image of this academic paper."

        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": user_query_text},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "systemInstruction": {"parts": [{"text": system_prompt_text}]},
        }

        headers = {"Content-Type": "application/json"}
        url = f"{BASE_API_URL}/{MODEL_NAME}:generateContent?key={API_KEY}"

        options = {
            "url": url,
            "method": "POST",
            "headers": headers,
            "data": json.dumps(payload),
            "timeout": 60
        }

        response = fetch_with_retry(url, options)
        result = response.json()

        candidate = result.get("candidates", [{}])[0]
        title = candidate.get("content", {}).get("parts", [{}])[0].get("text", "").strip()

        if not title:
            error_message = result.get("error", {}).get("message", "VLM returned no usable title.")
            raise Exception(error_message)

        # --- Basic cleanup ---
        title = title.replace("\r", " ").replace("\n", " ")
        title = " ".join(title.split())

        # Filename sanitization: replace colons for filesystem compatibility
        safe_print(f"[{threading.current_thread().name}] 3. Applying filesystem-safe character replacements...")
        title = title.replace(": ", " -- ").replace(":", "-")

        return title

    except Exception as e:
        safe_print(f"[{threading.current_thread().name}] [Error] Extraction failed ({base_filename}): {e}")
        return None
    finally:
        if doc:
            doc.close()


def extract_and_rename(file_path):
    """
    Extract title and rename file. Used for multi-threaded execution.
    """
    base_filename = os.path.basename(file_path)

    if not os.path.exists(file_path) or not file_path.lower().endswith(".pdf"):
        safe_print(f"[{threading.current_thread().name}] [Skip] File does not exist or is not a PDF: {base_filename}")
        return

    extracted_title = extract_title_from_pdf(file_path)

    if extracted_title:
        file_dir = os.path.dirname(file_path)

        # Remove strictly forbidden filesystem characters (/, \, ?, *, :, |, <, >, ")
        illegal_chars = r'[\\/?:*|<>"]'
        # Also strip leading/trailing dots (can hide files on some systems)
        safe_title = re.sub(illegal_chars, "", extracted_title).strip()
        safe_title = safe_title.rstrip(".").lstrip(".")

        if not safe_title:
            safe_print(f"[{threading.current_thread().name}] [Error] Title is empty after cleanup, cannot rename.")
            return

        new_filename = f"{safe_title}.pdf"
        new_file_path = os.path.join(file_dir, new_filename)

        try:
            os.rename(file_path, new_file_path)
            safe_print(f"[{threading.current_thread().name}] [Success] {base_filename} -> {new_filename}")
        except OSError as e:
            safe_print(f"[{threading.current_thread().name}] [Error] Cannot rename {base_filename}: {e}")

    safe_print("-" * 30)

def main():
    """Main: parallel batch file renaming"""
    if len(sys.argv) < 2:
        print("Usage: python rename_script.py <PDF1> [PDF2] ...")
        print("Example: python rename_script.py paper.pdf /path/to/*.pdf")
        return

    pdf_files = sys.argv[1:]
    max_workers = min(len(pdf_files), os.cpu_count() or 4, 8)

    print(f"\n--- AI Paper Renamer (Parallel) ---")
    print(f"Processing {len(pdf_files)} files with up to {max_workers} threads.\n")

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(extract_and_rename, pdf_files))

    end_time = time.time()
    print(f"\n--- Done ---")
    print(f"Total time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
```

### Option B: OpenAI-compatible API (Vercel AI Gateway, etc.)

```python
#!/usr/bin/env python3
import fitz  # PyMuPDF
import requests
import base64
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import threading
import re

# --- Configuration ---
# Replace these with your own settings
API_KEY = "YOUR_API_KEY"
BASE_API_URL = "https://ai-gateway.vercel.sh/v1"
MODEL_NAME = "google/gemini-2.5-flash-image"
COMPLETIONS_ENDPOINT = "/chat/completions"

# Thread-safe lock
print_lock = threading.Lock()

def safe_print(message):
    """Thread-safe print"""
    with print_lock:
        print(message)

def fetch_with_retry(url, options, max_retries=5):
    """API call with exponential backoff"""
    for i in range(max_retries):
        try:
            response = requests.request(**options)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            if i < max_retries - 1:
                delay = (2 ** i) * 1 + (time.time() % 1)
                safe_print(f"[Warning] API failed ({e}). Retry {i + 1}/{max_retries}, wait {delay:.2f}s")
                time.sleep(delay)
                continue
            raise

def extract_title_from_pdf(pdf_path):
    """Extract paper title from first page image via VLM."""
    base_filename = os.path.basename(pdf_path)
    safe_print(f"[{threading.current_thread().name}] Processing: {base_filename}")

    if not API_KEY or API_KEY == "YOUR_API_KEY":
        safe_print(f"[{threading.current_thread().name}] [Error] Missing API key. Update API_KEY in the script.")
        return None

    doc = None
    try:
        doc = fitz.open(pdf_path)
        page = doc[0]

        safe_print(f"[{threading.current_thread().name}] 1. Loading PDF and calculating crop area...")
        scale = 3.0
        matrix = fitz.Matrix(scale, scale)

        # Dynamic crop: find top text block
        text_blocks = page.get_text("dict")["blocks"]
        top_content_y_pdf_unit = float("inf")
        for block in text_blocks:
            if "lines" in block:
                for line in block["lines"]:
                    if line["spans"]:
                        top_content_y_pdf_unit = min(top_content_y_pdf_unit, line["bbox"][1])

        if top_content_y_pdf_unit == float("inf"):
            safe_print(f"[{threading.current_thread().name}] [Warning] No text detected, fallback crop.")
            top_content_y_pdf_unit = 0

        pdf_unit_buffer = 5
        start_y_pdf_unit = max(0, top_content_y_pdf_unit - pdf_unit_buffer)
        max_content_height_ratio = 0.40
        max_crop_height = int(page.rect.height * scale * max_content_height_ratio)

        crop_rect_pdf = fitz.Rect(
            page.rect.x0,
            start_y_pdf_unit,
            page.rect.x1,
            min(page.rect.y1, start_y_pdf_unit + (max_crop_height / scale))
        )

        pix = page.get_pixmap(matrix=matrix, clip=crop_rect_pdf)
        img_bytes = pix.tobytes("png")
        base64_image = base64.b64encode(img_bytes).decode("utf-8")

        safe_print(f"[{threading.current_thread().name}] Crop done. Image size: {pix.width}x{pix.height}.")

        # VLM API call (OpenAI compatible)
        system_prompt_text = (
            "You are a professional academic paper title extraction assistant. Only identify the title of the academic paper from the provided image."
            "Return the extracted title as plain text, **strictly ensuring the result is on a single line with no newline characters**."
            "**For words that appear in small caps or all caps in the original paper, normalize them to standard case (Title Case),**"
            "**unless the word is a clear and recognized all-caps acronym (e.g., 'NLP', 'AI', 'LLM', 'DNN', etc.),**"
            "in which case, preserve its uppercase form."
        )
        user_query_text = "Please extract the exact title from the first page image of this academic paper."

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt_text},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query_text},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}", "detail": "auto"}}
                    ]
                }
            ],
            "max_tokens": 100,
            "stream": False
        }

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {API_KEY}"
        }

        url = f"{BASE_API_URL}{COMPLETIONS_ENDPOINT}"
        options = {
            "url": url,
            "method": "POST",
            "headers": headers,
            "data": json.dumps(payload),
            "timeout": 60
        }

        response = fetch_with_retry(url, options)
        result = response.json()
        title = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip()

        if not title:
            raise Exception(f"API returned no title: {json.dumps(result, indent=2)}")

        # Cleanup
        title = title.replace("\r", " ").replace("\n", " ")
        title = " ".join(title.split())
        title = title.replace(": ", " -- ").replace(":", "-")

        return title

    except Exception as e:
        safe_print(f"[{threading.current_thread().name}] [Error] Extraction failed ({base_filename}): {e}")
        return None
    finally:
        if doc:
            doc.close()

def extract_and_rename(file_path):
    """Extract title and rename file"""
    base_filename = os.path.basename(file_path)
    if not os.path.exists(file_path) or not file_path.lower().endswith(".pdf"):
        safe_print(f"[{threading.current_thread().name}] [Skip] Not a PDF: {base_filename}")
        return

    extracted_title = extract_title_from_pdf(file_path)
    if extracted_title:
        file_dir = os.path.dirname(file_path)
        illegal_chars = r"[\\/?:*|<>\"\n\r\t]"
        safe_title = re.sub(illegal_chars, "", extracted_title).strip().strip(".")

        if not safe_title:
            safe_print(f"[{threading.current_thread().name}] [Error] Empty title after cleanup.")
            return

        new_filename = f"{safe_title}.pdf"
        new_file_path = os.path.join(file_dir, new_filename)
        try:
            os.rename(file_path, new_file_path)
            safe_print(f"[{threading.current_thread().name}] [Success] {base_filename} -> {new_filename}")
        except OSError as e:
            safe_print(f"[{threading.current_thread().name}] [Error] Rename failed: {e}")

    safe_print("-" * 30)

def main():
    if len(sys.argv) < 2:
        print("Usage: python rename_script.py <PDF1> [PDF2] ...")
        return

    pdf_files = sys.argv[1:]
    max_workers = min(len(pdf_files), os.cpu_count() or 4, 8)

    print(f"\n--- AI Paper Renamer ---")
    print(f"Processing {len(pdf_files)} files with up to {max_workers} threads.\n")
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(extract_and_rename, pdf_files))

    end_time = time.time()
    print(f"\n--- Done ---")
    print(f"Total time: {end_time - start_time:.2f}s")

if __name__ == "__main__":
    main()
```

---

## 4. (macOS) Create the Automator Quick Action

The following steps use macOS Automator and Finder. Skip to section 7 if you only need the command-line script.

Paste the following AppleScript directly into Automator when creating the Quick Action:

```applescript
tell application "Finder"
	set selectedItems to selection
	set pdfPaths to {}

	repeat with itemRef in selectedItems
		set itemPath to POSIX path of (itemRef as alias)
		if itemPath ends with ".pdf" or itemPath ends with ".PDF" then
			set end of pdfPaths to itemPath
		end if
	end repeat

	if (count of pdfPaths) > 0 then
		set scriptPath to POSIX path of (path to home folder) & "Library/Application Support/AI Paper Renamer/rename_script.py"
		set shellCmd to "/usr/bin/env python3 " & quoted form of scriptPath

		repeat with pdfPath in pdfPaths
			set shellCmd to shellCmd & " " & quoted form of pdfPath
		end repeat

		do shell script shellCmd
	end if
end tell
```

---

## 5. (macOS) Save the Quick Action

1. Open **Automator**
2. Choose **Quick Action**
3. At the top:
   - Workflow receives: **PDF files**
   - In: **Finder**
4. Add **Run AppleScript**
5. Paste the script from section 4
6. Save as: **AI Paper Renamer**

---

## 6. (macOS) Bind a keyboard shortcut

1. Open **System Settings → Keyboard → Keyboard Shortcuts → Services**
2. Expand **Files and Folders**, find **AI Paper Renamer**
3. Assign any unused shortcut (e.g., F13 or ⌃⌥⌘R)

Done: select PDF(s) in Finder and press your shortcut to rename.

---

## 7. Test & Troubleshooting

### 7.1 Test from Terminal

```sh
python3 "$HOME/Library/Application Support/AI Paper Renamer/rename_script.py" /path/to/your.pdf
```

### 7.2 Common issues

- **Missing API key**
  - Update `API_KEY` in the script.
- **Script not found**
  - Verify the path:
    `~/Library/Application Support/AI Paper Renamer/rename_script.py`
- **Missing dependencies**
  - Reinstall:
    `python3 -m pip install pymupdf requests`
