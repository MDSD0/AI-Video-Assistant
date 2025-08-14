import gradio as gr
from backend import VideoAssistantBackend
import dotenv
import os
import base64

with open("logo.png", "rb") as img_file:
    img_base64 = base64.b64encode(img_file.read()).decode("utf-8")


# Load environment variables
dotenv.load_dotenv()
if not os.environ.get("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY not found in .env file")

backend = VideoAssistantBackend(gemini_api_key=os.environ["GEMINI_API_KEY"])

def format_history(history):
    """Format the history into an HTML string for the context glossary."""
    if not history:
        return "<p>No history yet.</p>"
    html = ""
    for i in range(0, len(history), 3):
        video_num = i // 3 + 1
        html += f"<h3>Video {video_num}</h3>"
        if i < len(history):
            visual_context = history[i]['parts'][0]['text'].replace("Visual context: ", "")
            html += f"<p><strong>Visual Context:</strong> {visual_context}</p>"
        if i + 1 < len(history):
            user_query = history[i + 1]['parts'][0]['text'].replace("User query: ", "")
            html += f"<p><strong>User Query:</strong> {user_query}</p>"
        if i + 2 < len(history):
            assistant_response = history[i + 2]['parts'][0]['text']
            html += f"<p><strong>Assistant Response:</strong> {assistant_response}</p>"
    return html

with gr.Blocks(css="style.css", title="AI Video Assistant") as iface:
    # Header with vector image placeholder
    gr.Markdown(f"""
    <div class="header">
        <center>
            <img src="data:image/png;base64,{img_base64}" class="icon" style="height: 80px;" />
            <h1 style="margin: 0;">AI Video Assistant</h1>
            <p style="font-size: 1.1rem; color: #ccc;">Show me something and ask a question — I’ll respond based on what I see.</p>
        </center>
    </div> """)

    history_state = gr.State([])
    video_counter_state = gr.State(0)

    # Main Layout: Video left, Response right
    with gr.Row(elem_id="top-row"):
        with gr.Column(elem_id="left-panel"):
            video_input = gr.Video(label="Video", elem_id="video-box")
            process_button = gr.Button("Process Video", elem_id="process-btn")

        with gr.Column(elem_id="right-panel"):
            transcription_output = gr.Textbox(
                label="Your Question", interactive=False,
                lines=3, max_lines=4, show_copy_button=True, elem_id="question-box"
            )
            response_output = gr.Textbox(
                label="AI Response", interactive=False,
                lines=4, max_lines=6, show_copy_button=True, elem_id="response-box"
            )

    # Audio Section
    gr.Markdown('<div class="audio-label">Assistant’s Voice</div>')
    audio_output = gr.Audio(autoplay=True, elem_id="audio-player")

    # Context Glossary
    gr.Markdown("## Context Glossary")
    history_display = gr.HTML(value=format_history([]), elem_id="history-display")

    # Processing Logic
    def on_process(video, history, video_counter):
        audio_path, user_prompt, response_text, new_history, new_counter = backend.process_video(
            video, history, video_counter)
        history_html = format_history(new_history)
        return audio_path, user_prompt, response_text, new_history, new_counter, history_html

    process_button.click(
        fn=on_process,
        inputs=[video_input, history_state, video_counter_state],
        outputs=[audio_output, transcription_output, response_output, history_state, video_counter_state, history_display]
    )

if __name__ == "__main__":
    iface.launch()