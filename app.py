import os
import io
import uuid
import tempfile
import requests

from flask import (
    Flask,
    render_template,
    request,
    send_file,
    session,
    redirect,
    url_for
)

from fpdf import FPDF
from PIL import Image

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =================================================
# FLASK APP
# =================================================
app = Flask(__name__)


app.secret_key = "dev_secret_key"
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024


# =================================================
# SESSION INIT
# =================================================
@app.before_request
def ensure_session():
    if "sid" not in session:
        session["sid"] = str(uuid.uuid4())


# =================================================
# IN-MEMORY CACHE
# =================================================
DOCUMENT_CACHE = {}  # sid -> document chunks


# =================================================
# SAFE PDF BYTES (FIXES bytearray ERROR)
# =================================================
def pdf_to_bytes(pdf: FPDF) -> bytes:
    output = pdf.output(dest="S")

    if isinstance(output, str):
        return output.encode("latin-1")

    return bytes(output)  # handles bytearray / bytes


# =================================================
# TEXT → PDF
# =================================================
def create_pdf_from_text(text):
    pdf = FPDF()
    pdf.add_page()

    font_path = os.path.join(
        os.path.dirname(__file__),
        "fonts",
        "DejaVuSans.ttf"
    )

    if os.path.exists(font_path):
        pdf.add_font("DejaVu", "", font_path, uni=True)
        pdf.set_font("DejaVu", size=12)
    else:
        pdf.set_font("Arial", size=12)

    pdf.multi_cell(0, 8, text)

    return io.BytesIO(pdf_to_bytes(pdf))


# =================================================
# HOME
# =================================================
@app.route("/")
def home():
    return render_template("home.html")


# =================================================
# IMAGE → PDF (NO DISK USAGE)
# =================================================
@app.route("/image_to_pdf")
def image_to_pdf_form():
    return render_template("image.html")


@app.route("/upload_images", methods=["POST"])
def upload_images():
    images = request.files.getlist("images")
    if not images:
        return "No images uploaded", 400

    pdf = FPDF(unit="mm", format="A4")
    PAGE_W, PAGE_H, MARGIN = 210, 297, 15

    for img_file in images:
        img = Image.open(img_file).convert("RGB")

        img_w, img_h = img.size
        dpi = img.info.get("dpi", (72,))[0]

        w_mm = img_w * 25.4 / dpi
        h_mm = img_h * 25.4 / dpi

        scale = min(
            (PAGE_W - 2 * MARGIN) / w_mm,
            (PAGE_H - 2 * MARGIN) / h_mm
        )

        pdf.add_page()

        img_buffer = io.BytesIO()
        img.save(img_buffer, format="JPEG", quality=85)
        img_buffer.seek(0)

        pdf.image(
            img_buffer,
            x=(PAGE_W - w_mm * scale) / 2,
            y=(PAGE_H - h_mm * scale) / 2,
            w=w_mm * scale,
            h=h_mm * scale
        )

    return send_file(
        io.BytesIO(pdf_to_bytes(pdf)),
        mimetype="application/pdf",
        as_attachment=True,
        download_name="images.pdf"
    )


# =================================================
# TEXT → PDF ROUTES
# =================================================
@app.route("/text-to-pdf")
def text_to_pdf_form():
    return render_template("text.html")


@app.route("/generate_pdf", methods=["POST"])
def generate_pdf():
    text = request.form.get("paragraph", "")
    name = request.form.get("pdf_name", "document")

    pdf_buffer = create_pdf_from_text(text)

    return send_file(
        pdf_buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=f"{name}.pdf"
    )


# =================================================
# PDF Q&A CORE
# =================================================
def load_pdf_and_split(pdf_file):
    # Create temp file WITHOUT delete=True
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_path = tmp.name
    tmp.close()  # 🔥 IMPORTANT (releases Windows lock)

    # Save uploaded PDF
    pdf_file.save(tmp_path)

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            chunk_overlap=100
        )

        return splitter.split_documents(pages)

    finally:
        # Clean up temp file manually
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def ask_question_from_pdf(docs, question):
    q_words = set(question.lower().split())
    scored = []

    for d in docs:
        score = sum(w in d.page_content.lower() for w in q_words)
        scored.append((score, d.page_content))

    scored.sort(reverse=True)
    context = "\n\n".join(c for s, c in scored[:3] if s > 0)

    if not context:
        return "Answer not found in the document."

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [
                {
                    "role": "user",
                    "content": f"""
Context:
{context}

Question:
{question}

Answer:
"""
                }
            ],
            "temperature": 0.2,
            "max_tokens": 400
        }
    )

    return response.json()["choices"][0]["message"]["content"].strip()


# =================================================
# PDF Q&A ROUTES
# =================================================
@app.route("/pdf-qa", methods=["GET"])
def pdf_qa():
    sid = session["sid"]

    answer = session.pop("last_answer", None)
    question = session.pop("last_question", None)
    uploaded_pdf = session.get("uploaded_pdf_name")

    if answer is None:
        DOCUMENT_CACHE.pop(sid, None)
        session.pop("uploaded_pdf_name", None)
        uploaded_pdf = None

    return render_template(
        "pdf_qa.html",
        question=question,
        answer=answer,
        uploaded_pdf=uploaded_pdf
    )


@app.route("/ask_pdf", methods=["POST"])
def ask_pdf():
    sid = session["sid"]

    question = request.form.get("question")
    pdf_file = request.files.get("pdf_file")

    if sid not in DOCUMENT_CACHE:
        if not pdf_file or not pdf_file.filename:
            return render_template(
                "pdf_qa.html",
                error="Please upload a PDF first."
            )

        DOCUMENT_CACHE[sid] = load_pdf_and_split(pdf_file)
        session["uploaded_pdf_name"] = pdf_file.filename

    answer = ask_question_from_pdf(
        DOCUMENT_CACHE[sid],
        question
    )

    session["last_answer"] = answer
    session["last_question"] = question

    return redirect(url_for("pdf_qa"))


# =================================================
# LOCAL RUN
# =================================================
if __name__ == "__main__":
    app.run(debug=True)
