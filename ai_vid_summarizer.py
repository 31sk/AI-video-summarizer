import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi
import moviepy as mp
import speech_recognition as sr
from transformers import pipeline
from pydub import AudioSegment
import os
import tempfile
import requests
from pytube import YouTube

# --- Styling Constants ---
PRIMARY_COLOR = "#6a5acd"  # Slate Blue
SECONDARY_COLOR = "#3cb371"  # Sea Green
WARNING_COLOR = "#f0ad4e"  # Warning Orange
ERROR_COLOR = "#d9534f"  # Error Red
BACKGROUND_COLOR_1 = "#f5f5dc"  # Beige
BACKGROUND_COLOR_2 = "#fffaf0"  # Ivory

# --- Streamlit Theme Configuration ---
st.markdown(
    f"""
    <style>
    .reportview-container .main {{
        background: linear-gradient(to bottom, {BACKGROUND_COLOR_1}, {BACKGROUND_COLOR_2});
        color: {PRIMARY_COLOR};
    }}
    .stButton>button {{
        color: white;
        background-color: {PRIMARY_COLOR};
        border: 2px solid {SECONDARY_COLOR};
        border-radius: 5px;
        padding: 0.5em 1em;
    }}
    .stButton>button:hover {{
        background-color: {SECONDARY_COLOR};
        color: white;
        border-color: {PRIMARY_COLOR};
    }}
    .stTextInput>label, .stFileUploader>label, .stRadio>label {{
        color: {PRIMARY_COLOR};
    }}
    h1, h2, h3, h4, h5, h6 {{
        color: {PRIMARY_COLOR};
        font-family: 'Arial', sans-serif;
    }}
    .streamlit-expanderHeader {{
        color: {PRIMARY_COLOR};
    }}
    .stSpinner > div > div {{
        color: {SECONDARY_COLOR};
    }}
    .stSuccess {{
        color: {SECONDARY_COLOR};
    }}
    .stWarning {{
        color: {WARNING_COLOR};
    }}
    .stError {{
        color: {ERROR_COLOR};
    }}
    /* Add a subtle shadow to elements */
    .css-1lcbmhc, .css-ke9vvo, .css-r698ls {{
        box-shadow: 0 0.15rem 0.5rem rgba(0, 0, 0, 0.15);
    }}
    /* Add padding around elements for better spacing */
    .stRadio, .stTextInput, .stFileUploader, .stButton {{
        padding-bottom: 15px;
    }}

    </style>
    """,
    unsafe_allow_html=True,
)


def get_youtube_transcript(video_id):
    """Extracts transcript from a YouTube video."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        text = " ".join([entry['text'] for entry in transcript])
        return text
    except Exception as e:
        st.error(f":red[Error getting YouTube transcript:] {e}")
        st.error(f":red[Detailed Error:] {e}")
        return None


def transcribe_audio(audio_file):
    """Transcribes audio from a video file."""
    r = sr.Recognizer()
    full_text = ""
    try:
        audio = AudioSegment.from_file(audio_file)
        chunk_length_ms = 60000  # split into 60-second chunks
        chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

        temp_dir = tempfile.mkdtemp()

        for i, chunk in enumerate(chunks):
            chunk_file = os.path.join(temp_dir, f"chunk{i}.wav")
            chunk.export(chunk_file, format="wav")

            with sr.AudioFile(chunk_file) as source:
                audio_data = r.record(source)

            try:
                text = r.recognize_google(audio_data)
                full_text += text + " "
            except sr.UnknownValueError:
                st.warning(f":orange[Chunk {i}: Speech Recognition could not understand audio]")
            except sr.RequestError as e:
                st.error(f":red[Chunk {i}: Could not request results from Speech Recognition service;] {e}")

            os.remove(chunk_file)

        os.rmdir(temp_dir)

        return full_text

    except Exception as e:
        st.error(f":red[Error transcribing audio:] {e}")
        return None


def summarize_text(text):
    """Summarizes the given text using a transformer model."""
    if not text or len(text) == 0:
        st.warning(":orange[No text available to summarize.]")
        return None

    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    try:
        summary_output = summarizer(text, max_length=130, min_length=30, do_sample=False)
        if summary_output and len(summary_output) > 0:
            summary = summary_output[0]['summary_text']
            return summary
        else:
            st.warning(":orange[Summarization failed: No summary returned.]")
            return None

    except Exception as e:
        st.error(f":red[Error summarizing text:] {e}")
        st.error(f":red[Summarization Error:] {e}")
        return None


def process_video(video_file):
    """Processes a video file to extract audio and transcribe it."""
    try:
        video = mp.VideoFileClip(video_file)
        audio = video.audio
        temp_audio_file = "temp_audio.wav"
        audio.write_audiofile(temp_audio_file, codec='pcm_s16le')

        transcript = transcribe_audio(temp_audio_file)
        video.close()
        audio.close()
        os.remove(temp_audio_file)
        return transcript
    except Exception as e:
        st.error(f":red[Error processing video:] {e}")
        return None


def download_youtube_video(url):
    """Downloads a YouTube video to a temporary file."""
    try:
        yt = YouTube(url)
        video = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
        if video:
            temp_dir = tempfile.mkdtemp()
            video_filepath = os.path.join(temp_dir, f"{yt.title}.mp4")
            video.download(output_path=temp_dir)
            return video_filepath
        else:
            st.error(":red[No suitable video stream found.]")
            return None
    except Exception as e:
        st.error(f":red[Error downloading YouTube video:] {e}")
        return None


def main():
    """Main function to run the Streamlit app."""
    st.title("Video/YouTube Summarizer")
    st.markdown("Upload a video or paste a YouTube link to get a summary!")

    input_type = st.radio("Input Type:", ("YouTube Link", "Video File"))

    if input_type == "YouTube Link":
        youtube_url = st.text_input("YouTube Video URL:")
        if youtube_url:
            try:
                video_id = youtube_url.split("watch?v=")[1].split("&")[0]
            except IndexError:
                st.error(":red[Invalid YouTube URL]")
                return

            transcript = get_youtube_transcript(video_id)
            if transcript:
                with st.spinner(":green[Summarizing...]") as spinner:
                    summary = summarize_text(transcript)
                if summary:
                    st.subheader(":green[Summary:]")
                    st.write(summary)
            else:
                st.warning(":orange[Could not retrieve transcript. Trying to summarize by downloading the video.]")
                temp_file = download_youtube_video(youtube_url)
                if temp_file:
                    transcript = process_video(temp_file)

                    if transcript:
                        with st.spinner(":green[Summarizing...]") as spinner:
                            summary = summarize_text(transcript)
                        if summary:
                            st.subheader(":green[Summary:]")
                            st.write(summary)
                    else:
                        st.warning(":orange[Could not process video to get transcript.]")

                    try:
                        os.remove(temp_file)
                        os.rmdir(os.path.dirname(temp_file))
                    except:
                        pass
                else:
                    st.warning(":orange[Could not download YouTube video.]")

    elif input_type == "Video File":
        video_file = st.file_uploader("Upload Video", type=["mp4", "avi", "mov", "mkv"])
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=f".{video_file.name.split('.')[-1]}")
            tfile.write(video_file.read())
            video_filepath = tfile.name
            st.video(video_filepath)

            with st.spinner(":green[Processing video and transcribing...]") as spinner:
                transcript = process_video(video_filepath)

            if transcript:
                with st.spinner(":green[Summarizing...]") as spinner:
                    summary = summarize_text(transcript)
                if summary:
                    st.subheader(":green[Summary:]")
                    st.write(summary)
            else:
                st.warning(":orange[Could not process video or transcribe audio.]")
            try:
                tfile.close()
                os.unlink(video_filepath)
            finally:
                pass


if __name__ == "__main__":
    main()