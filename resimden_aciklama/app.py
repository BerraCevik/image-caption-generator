import customtkinter as ctk
from PIL import Image
from tkinter import filedialog, messagebox
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    CLIPProcessor,
    CLIPModel,
    AutoProcessor,          # GIT için
    AutoModelForCausalLM,   # GIT için
    VisionEncoderDecoderModel,  # ViT-GPT2 için
    ViTImageProcessor,          # ViT-GPT2 için
    AutoTokenizer,              # ViT-GPT2 için
)

from deep_translator import GoogleTranslator
from gtts import gTTS
import torch, os, time, pygame, tempfile

# ---------- MODEL ----------

device = "cuda" if torch.cuda.is_available() else "cpu"

# BLIP (caption)
MODEL_NAME = "Salesforce/blip-image-captioning-base"
processor = BlipProcessor.from_pretrained(MODEL_NAME)
model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)

# CLIP (görsel + metin benzerliği)
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(device)

# GIT (BLIP'ten farklı, güçlü bir caption modeli)
# COCO'ya ince ayarlı versiyonu kullanıyoruz:
GIT_MODEL_NAME = "microsoft/git-base-coco"

git_processor = AutoProcessor.from_pretrained(GIT_MODEL_NAME)
git_dtype = torch.float16 if device == "cuda" else torch.float32

git_model = AutoModelForCausalLM.from_pretrained(
    GIT_MODEL_NAME,
    torch_dtype=git_dtype
).to(device)
git_model.eval()

# ViT-GPT2 (BLIP & GIT’ten farklı encoder–decoder caption modeli)
VIT_MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

vit_model = VisionEncoderDecoderModel.from_pretrained(VIT_MODEL_NAME).to(device)
vit_processor = ViTImageProcessor.from_pretrained(VIT_MODEL_NAME)
vit_tokenizer = AutoTokenizer.from_pretrained(VIT_MODEL_NAME)
vit_model.eval()

last_caption = ""       # Kullanıcıya gösterilen (TR veya EN)
last_caption_en = ""    # BLIP'in ham İngilizce çıktısı


# ---------- FONKSİYONLAR ----------

def generate_caption(img):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=30)
    return processor.decode(out[0], skip_special_tokens=True).strip()


def translate_text(text, lang):
    if lang == "Türkçe":
        return GoogleTranslator(source="en", target="tr").translate(text)
    return text


def make_tts(text):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
        filename = fp.name
    tts = gTTS(text=text, lang="tr")
    tts.save(filename)
    pygame.mixer.init()
    pygame.mixer.music.load(filename)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        time.sleep(0.1)
    pygame.mixer.quit()
    try:
        os.remove(filename)
    except:
        pass


def open_image():
    global last_caption, last_caption_en
    path = filedialog.askopenfilename(
        title="Görsel Seç",
        filetypes=[("Resimler", "*.jpg *.jpeg *.png")]
    )
    if not path:
        return

    img = Image.open(path).convert("RGB")
    img.thumbnail((300, 300))
    img_ctk = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))
    img_label.configure(image=img_ctk, text="")
    img_label.image = img_ctk
    img_label.file = path

    last_caption = ""
    last_caption_en = ""

    # metin kutularını temizle (4 sekme: BLIP, CLIP, GIT, ViT-GPT2)
    for box in [box_blip, box_clip, box_git, box_vit]:
        box.configure(state="normal")
        box.delete("1.0", "end")
        box.configure(state="disabled")

    btn_listen_blip.configure(state="disabled")


def analyze_blip():
    global last_caption, last_caption_en
    if not hasattr(img_label, "file"):
        messagebox.showerror("Hata", "Önce bir görsel seçin.")
        return

    img = Image.open(img_label.file).convert("RGB")
    lang = lang_blip.get()

    # Her zaman önce İngilizce caption üret (modelin doğal dili)
    caption_en = generate_caption(img)
    last_caption_en = caption_en

    # Arayüz diline göre gösterilecek cümleyi seç
    caption_display = translate_text(caption_en, lang)

    box_blip.configure(state="normal")
    box_blip.delete("1.0", "end")
    box_blip.insert("end", caption_display)
    box_blip.configure(state="disabled")

    last_caption = caption_display
    btn_listen_blip.configure(state="normal")


def analyze_clip():
    """
    CLIP ile: kullanıcıdan alınan metin etiketleri (virgülle ayrılmış),
    görüntü ile eşleştiriliyor. Model bu etiketlerden hangisinin
    görsele daha çok uyduğunu olasılıksal olarak söylüyor.
    """
    if not hasattr(img_label, "file"):
        messagebox.showerror("Hata", "Önce bir görsel seçin.")
        return

    labels_str = entry_clip_labels.get().strip()
    if not labels_str:
        messagebox.showerror("Hata", "Lütfen en az bir metin etiketi girin.")
        return

    texts = [t.strip() for t in labels_str.split(",") if t.strip()]
    if not texts:
        messagebox.showerror("Hata", "Geçerli bir metin etiketi listesi girin.")
        return

    img = Image.open(img_label.file).convert("RGB")

    inputs = clip_processor(
        text=texts,
        images=img,
        return_tensors="pt",
        padding=True
    ).to(device)

    with torch.inference_mode():
        outputs = clip_model(**inputs)
        logits_per_image = outputs.logits_per_image[0]  # [N]
        probs = logits_per_image.softmax(dim=0)

    values, indices = probs.sort(descending=True)
    lines = ["CLIP'e göre en olası eşleşmeler:\n"]
    top_k = min(5, len(texts))
    for i in range(top_k):
        idx = indices[i].item()
        label = texts[idx]
        p = values[i].item() * 100
        lines.append(f"{i+1}. {label}  ({p:.1f}%)")

    result = "\n".join(lines)

    box_clip.configure(state="normal")
    box_clip.delete("1.0", "end")
    box_clip.insert("end", result)
    box_clip.configure(state="disabled")


def analyze_git():
    """
    GIT (microsoft/git-base-coco) ile caption üretir.
    BLIP'ten farklı bir mimari (CLIP + Transformer decoder).
    """
    if not hasattr(img_label, "file"):
        messagebox.showerror("Hata", "Önce bir görsel seçin.")
        return

    img = Image.open(img_label.file).convert("RGB")
    lang = lang_blip.get()  # "Türkçe" / "English"

    # GIT pipeline: AutoProcessor + AutoModelForCausalLM
    pixel_values = git_processor(images=img, return_tensors="pt").pixel_values.to(device)

    with torch.inference_mode():
        generated_ids = git_model.generate(
            pixel_values=pixel_values,
            max_length=50,
            num_beams=4,
            early_stopping=True
        )

    caption_en = git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

    if lang == "Türkçe":
        try:
            caption_display = GoogleTranslator(source="en", target="tr").translate(caption_en)
        except Exception:
            caption_display = caption_en
    else:
        caption_display = caption_en

    box_git.configure(state="normal")
    box_git.delete("1.0", "end")
    box_git.insert("end", caption_display)
    box_git.configure(state="disabled")


def analyze_vit():
    """
    ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning) ile caption üretir.
    BLIP ve GIT'ten farklı klasik encoder–decoder yapısı kullanır.
    """
    if not hasattr(img_label, "file"):
        messagebox.showerror("Hata", "Önce bir görsel seçin.")
        return

    img = Image.open(img_label.file).convert("RGB")
    lang = lang_blip.get()  # "Türkçe" / "English"

    pixel_values = vit_processor(images=img, return_tensors="pt").pixel_values.to(device)

    with torch.inference_mode():
        output_ids = vit_model.generate(
            pixel_values,
            max_length=64,
            num_beams=4,
            early_stopping=True
        )

    caption_en = vit_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

    if lang == "Türkçe":
        try:
            caption_display = GoogleTranslator(source="en", target="tr").translate(caption_en)
        except Exception:
            caption_display = caption_en
    else:
        caption_display = caption_en

    box_vit.configure(state="normal")
    box_vit.delete("1.0", "end")
    box_vit.insert("end", caption_display)
    box_vit.configure(state="disabled")


def play_audio():
    if not last_caption:
        return
    make_tts(last_caption)


# ---------- ARAYÜZ ----------

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

root = ctk.CTk()
root.title("🧠 Multi-Model Görsel Açıklama Aracı")
root.geometry("1350x800")

# -------- sol panel (görsel) --------
left = ctk.CTkFrame(root, corner_radius=15)
left.pack(side="left", fill="y", padx=15, pady=15)

title = ctk.CTkLabel(left, text="🖼️ Görsel Yükleme", font=ctk.CTkFont(size=18, weight="bold"))
title.pack(pady=15)

btn_img = ctk.CTkButton(left, text="📁 Görsel Seç", command=open_image)
btn_img.pack(pady=10)

img_label = ctk.CTkLabel(
    left,
    text="Henüz görsel seçilmedi",
    width=300,
    height=300,
    fg_color="#333333",
    corner_radius=10
)
img_label.pack(pady=20)

info = ctk.CTkLabel(
    left,
    text=(
        "BLIP: kısa açıklama üretir\n"
        "CLIP: metin etiketleriyle görseli eşleştirir\n"
        "GIT: BLIP'ten farklı, güçlü bir caption modeli\n"
        "ViT-GPT2: klasik encoder–decoder caption modeli (BLIP & GIT’ten farklı)."
    ),
    font=ctk.CTkFont(size=13)
)
info.pack(pady=10)

# -------- sağ panel: TABVIEW ile 4 sekme --------
right = ctk.CTkTabview(root, width=900)
right.pack(side="right", fill="both", expand=True, padx=15, pady=15)

tab_blip = right.add("BLIP")
tab_clip = right.add("CLIP")
tab_git_tab = right.add("GIT Caption")
tab_vit_tab = right.add("ViT-GPT2 Caption")

# ----- BLIP sekmesi -----
ctk.CTkLabel(
    tab_blip,
    text="🧠 BLIP (Caption Generation)",
    font=ctk.CTkFont(size=15, weight="bold")
).pack(anchor="w", padx=10, pady=5)

lang_blip = ctk.StringVar(value="Türkçe")
ctk.CTkOptionMenu(
    tab_blip,
    values=["Türkçe", "English"],
    variable=lang_blip,
    width=120
).pack(anchor="w", padx=10, pady=5)

ctk.CTkButton(
    tab_blip,
    text="Açıklama Üret",
    command=analyze_blip,
    fg_color="#1565C0"
).pack(anchor="w", padx=10, pady=5)

btn_listen_blip = ctk.CTkButton(
    tab_blip,
    text="🎧 Sesli Oynat (TR)",
    width=140,
    command=play_audio,
    fg_color="#2E7D32",
    state="disabled"
)
btn_listen_blip.pack(anchor="w", padx=10, pady=5)

box_blip = ctk.CTkTextbox(tab_blip, height=200, state="disabled")
box_blip.pack(padx=10, pady=5, fill="both", expand=True)

# ----- CLIP sekmesi -----
ctk.CTkLabel(
    tab_clip,
    text="🎯 CLIP (Image–Text Matching)",
    font=ctk.CTkFont(size=15, weight="bold")
).pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(
    tab_clip,
    text="Virgülle ayırarak metin etiketleri girin (İngilizce olması tercih edilir):",
    text_color="lightgray",
    font=ctk.CTkFont(size=13)
).pack(anchor="w", padx=10)

entry_clip_labels = ctk.CTkEntry(tab_clip, width=600)
entry_clip_labels.insert(0, "a photo of a dog, a photo of a cat, a photo of a person")
entry_clip_labels.pack(anchor="w", padx=10, pady=5)

ctk.CTkButton(
    tab_clip,
    text="CLIP ile Analiz Et",
    command=analyze_clip,
    fg_color="#8E24AA"
).pack(anchor="w", padx=10, pady=5)

box_clip = ctk.CTkTextbox(tab_clip, height=250, state="disabled")
box_clip.pack(padx=10, pady=5, fill="both", expand=True)

# ----- GIT sekmesi -----
ctk.CTkLabel(
    tab_git_tab,
    text="🧠 GIT (microsoft/git-base-coco)",
    font=ctk.CTkFont(size=15, weight="bold")
).pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(
    tab_git_tab,
    text=(
        "Bu sekmede BLIP'ten farklı bir caption modeli kullanılıyor (GIT).\n"
        "Model, görseli okuyup İngilizce açıklama üretiyor; arayüz diline göre Türkçeye çevriliyor."
    ),
    text_color="lightgray",
    font=ctk.CTkFont(size=13)
).pack(anchor="w", padx=10)

ctk.CTkButton(
    tab_git_tab,
    text="GIT ile Açıklama Üret",
    command=analyze_git,
    fg_color="#F4511E"
).pack(anchor="w", padx=10, pady=5)

box_git = ctk.CTkTextbox(tab_git_tab, height=250, state="disabled")
box_git.pack(padx=10, pady=5, fill="both", expand=True)

# ----- ViT-GPT2 sekmesi -----
ctk.CTkLabel(
    tab_vit_tab,
    text="🧠 ViT-GPT2 (nlpconnect/vit-gpt2-image-captioning)",
    font=ctk.CTkFont(size=15, weight="bold")
).pack(anchor="w", padx=10, pady=5)

ctk.CTkLabel(
    tab_vit_tab,
    text=(
        "Bu sekmede BLIP ve GIT'ten farklı klasik encoder–decoder caption modeli (ViT-GPT2) kullanılıyor.\n"
        "Model, görseli okuyup İngilizce açıklama üretiyor; arayüz diline göre Türkçeye çevriliyor."
    ),
    text_color="lightgray",
    font=ctk.CTkFont(size=13)
).pack(anchor="w", padx=10)

ctk.CTkButton(
    tab_vit_tab,
    text="ViT-GPT2 ile Açıklama Üret",
    command=analyze_vit,
    fg_color="#FF9800"
).pack(anchor="w", padx=10, pady=5)

box_vit = ctk.CTkTextbox(tab_vit_tab, height=250, state="disabled")
box_vit.pack(padx=10, pady=5, fill="both", expand=True)

footer = ctk.CTkLabel(root, text="Hazırlayan: brr zynp", font=ctk.CTkFont(size=12, slant="italic"))
footer.pack(pady=5)

root.mainloop()
