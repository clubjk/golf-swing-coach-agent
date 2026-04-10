import os
import gradio as gr
from openai import OpenAI

# Place don.jpg and peggy.jpg in the same folder as this script
# for character portraits. Emoji placeholders used if not found.
DON_IMG  = "don.jpg"  if os.path.exists("don.jpg")  else None
PEGGY_IMG = "peggy.jpg" if os.path.exists("peggy.jpg") else None

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ── System prompts ──────────────────────────────────────────────────────────

DON_SYSTEM = """You are Donald Francis Draper, Creative Director at Sterling Cooper Draper Pryce.
You are brilliant, elegant, deeply cynical, and poetic. Extremely difficult to please. You speak in short, powerful sentences with 1960s swagger.
You sell feelings, not products. You see the human truth behind every desire.

When Peggy presents a slogan:
• If it is truly great — praise her lavishly but coolly. Tell her this is the one. End your response with the single word APPROVED on its own line.
• If it is not great — be sharp, honest, and precise. Tell her exactly why it fails (too literal, generic, clever-for-clever's-sake, no emotional truth, etc.).
• Always give her clear, actionable direction on how to make it better. Push her to go deeper.
Never break character. Never be polite just to be nice."""

PEGGY_SYSTEM = """You are Peggy Olson, junior copywriter at Sterling Cooper Draper Pryce.
You are ambitious, talented, and still proving yourself. You respect Don deeply and learn fast.
Present each slogan clearly, followed by a short, confident rationale (1-2 sentences max).
When Don gives feedback, acknowledge it respectfully, then come back stronger with an improved version based exactly on his notes.
Stay completely in character."""

# ── Agent helper ─────────────────────────────────────────────────────────────

def get_response(system_prompt, messages):
    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": system_prompt}] + messages,
        temperature=0.78,
        max_tokens=450,
    )
    return completion.choices[0].message.content.strip()

# ── Meeting generator ─────────────────────────────────────────────────────────

def run_meeting(product, max_rounds):
    if not product.strip():
        product = "Sunkist oranges and soda"

    history = []
    messages = [{"role": "user", "content": f"Product/Business: {product}\n\nPeggy, bring me your best slogan for this product or business."}]

    for round_num in range(1, int(max_rounds) + 1):
        # Peggy presents
        peggy_reply = get_response(PEGGY_SYSTEM, messages)
        messages.append({"role": "assistant", "content": peggy_reply})
        history.append({"role": "user", "content": f"*— Round {round_num} —*\n\n{peggy_reply}"})
        yield history, ""

        # Don reviews
        don_reply = get_response(DON_SYSTEM, messages)
        messages.append({"role": "assistant", "content": don_reply})
        history.append({"role": "assistant", "content": don_reply})

        if "APPROVED" in don_reply:
            history.append({"role": "assistant", "content": "✅ **Meeting adjourned. That's the one.**"})
            yield history, peggy_reply
            return

        yield history, ""

    yield history, ""

# ── Character card HTML ───────────────────────────────────────────────────────

def character_card(name, title, emoji, img_path):
    if img_path:
        portrait = (
            f'<img src="/gradio_api/file={img_path}" '
            f'style="width:130px;height:170px;object-fit:cover;object-position:top;'
            f'border:3px solid #c9a84c;display:block;margin:0 auto;">'
        )
    else:
        portrait = (
            f'<div style="width:130px;height:170px;background:#1a1a2e;border:3px solid #c9a84c;'
            f'display:flex;align-items:center;justify-content:center;'
            f'font-size:5em;margin:0 auto;">{emoji}</div>'
        )
    return f"""
    <div style="text-align:center;padding:16px 8px;">
        {portrait}
        <div style="color:#c9a84c;font-family:Georgia,serif;font-size:1.05em;
                    font-weight:bold;margin-top:12px;letter-spacing:.05em;">{name}</div>
        <div style="color:#7a7a8a;font-family:Georgia,serif;font-style:italic;
                    font-size:0.8em;margin-top:4px;">{title}</div>
    </div>
    """

# ── CSS ───────────────────────────────────────────────────────────────────────

CSS = """
body, .gradio-container {
    background: #0c0c18 !important;
    font-family: Georgia, 'Times New Roman', serif !important;
}
#title {
    text-align: center;
    color: #c9a84c;
    letter-spacing: .2em;
    font-size: 1.7em;
    font-weight: bold;
    margin-bottom: 2px;
    padding-top: 16px;
}
#subtitle {
    text-align: center;
    color: #5a5a6a;
    font-style: italic;
    font-size: .9em;
    margin-bottom: 16px;
}
.divider {
    border: none;
    border-top: 1px solid #c9a84c44;
    margin: 12px 0;
}
/* Prompt line */
#prompt-line {
    text-align: center;
    color: #c9a84c;
    font-family: Georgia, serif;
    font-style: italic;
    font-size: 1.3em;
    font-weight: bold;
    margin: 14px 0 18px 0;
    letter-spacing: .03em;
}
/* Brief input */
#brief-input textarea {
    background: #12122a !important;
    color: #e8dcc8 !important;
    border: 1px solid #c9a84c55 !important;
    font-family: Georgia, serif !important;
}
#brief-input label {
    color: #c9a84c !important;
    font-family: Georgia, serif !important;
}
/* Rounds slider */
#rounds-slider label {
    color: #c9a84c !important;
    font-family: Georgia, serif !important;
}
/* Run button */
#run-btn {
    background: #c9a84c !important;
    color: #0c0c18 !important;
    font-family: Georgia, serif !important;
    font-weight: bold !important;
    letter-spacing: .12em !important;
    border: none !important;
    font-size: 1em !important;
}
#run-btn:hover {
    background: #e0c060 !important;
}
/* Approved box */
#approved-box textarea {
    background: #0c180c !important;
    color: #6fcf6f !important;
    border: 1px solid #4caf5055 !important;
    font-family: Georgia, serif !important;
    font-size: 1.05em !important;
    font-style: italic !important;
}
#approved-box label {
    color: #4caf50 !important;
    font-family: Georgia, serif !important;
}
"""

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Sterling Cooper Draper Pryce") as demo:

    gr.HTML('<div id="title">STERLING COOPER DRAPER PRYCE</div>')
    gr.HTML('<div id="subtitle">Creative Meeting Room &mdash; New York City, 1963</div>')
    gr.HTML('<hr class="divider">')

    with gr.Row():
        with gr.Column(scale=1, min_width=160):
            gr.HTML(character_card("Peggy Olson", "Junior Copywriter", "👩", PEGGY_IMG))

        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                value=[],
                avatar_images=(PEGGY_IMG, DON_IMG),
                height=480,
                show_label=False,
                render_markdown=True,
                layout="bubble",
            )

        with gr.Column(scale=1, min_width=160):
            gr.HTML(character_card("Don Draper", "Creative Director", "🤵", DON_IMG))

    gr.HTML('<hr class="divider">')
    gr.HTML('<div id="prompt-line">What product or business would you like the Mad Men creative team to <em>crush?</em> Enter it below and let Don &amp; Peggy work their magic.</div>')

    with gr.Row():
        brief_input = gr.Textbox(
            placeholder="e.g. Sunkist, Kodak cameras, American Airlines...",
            label="Product or Business Name",
            elem_id="brief-input",
            scale=4,
        )
        rounds_slider = gr.Slider(
            minimum=2, maximum=12, value=8, step=1,
            label="Max Rounds",
            elem_id="rounds-slider",
            scale=1,
        )

    with gr.Row():
        run_btn = gr.Button("▶  START THE MEETING", variant="primary", elem_id="run-btn")

    approved_box = gr.Textbox(
        label="✅  Approved Slogan",
        interactive=False,
        placeholder="Waiting for Don's approval...",
        elem_id="approved-box",
    )

    run_btn.click(
        fn=run_meeting,
        inputs=[brief_input, rounds_slider],
        outputs=[chatbot, approved_box],
    )

    brief_input.submit(
        fn=run_meeting,
        inputs=[brief_input, rounds_slider],
        outputs=[chatbot, approved_box],
    )

if __name__ == "__main__":
    demo.launch(allowed_paths=["."], css=CSS)
