# -*- coding: utf-8 -*-
"""Video_Moderation.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1bXKkA0Jk4Qjoag7KxaPKn6KorABKyodP
"""

# !pip install openai-whisper pydub

import os
import re
from moviepy.editor import VideoFileClip, concatenate_videoclips, AudioFileClip
import whisper
from pydub import AudioSegment

# Define harmful words for detection
harmful_words = {
    " stupid ", "bastards", "idiot", "dumb", "fool", "nasty", "fucking", "shit", "stupid.", " fucking ",
    "porn ", "Sex ", "pornography ", "stupider" , "stupid bastards ", "really stupid", "fucking stupid" , "fully shit",
    "how stupid is" , "stupider than that" , "not stupid" , "fucking nuts" 
}

def extract_audio(video_path):
    """
    Extract audio from video file.

    :param video_path: Path to the input video file
    :return: Path to the extracted audio file
    """
    video = VideoFileClip(video_path)
    audio_path = "extracted_audio.wav"
    video.audio.write_audiofile(audio_path)
    video.close()
    return audio_path

def transcribe_audio_whisper(audio_path):
    """
    Transcribe audio using Whisper model.

    :param audio_path: Path to the audio file
    :return: Transcribed text from the audio
    """
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

def convert_audio_to_text_with_timestamps_whisper(audio_path, transcribed_text):
    """
    Use the Whisper-transcribed text and assign timestamps in 2-second intervals.

    :param audio_path: Path to the audio file
    :param transcribed_text: The text transcribed by the Whisper model
    :return: List of tuples containing (timestamp, transcribed_text)
    """
    # Load the audio file
    audio = AudioSegment.from_wav(audio_path)

    # Split the audio into 2-second chunks
    chunk_length_ms = 2000
    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]

    transcription = []

    # Split the transcribed text into words
    words = transcribed_text.split()

    # Calculate the number of words per chunk
    chunk_size = len(words) // len(chunks) if len(chunks) > 0 else len(words)

    for i, chunk in enumerate(chunks):
        # Get the text corresponding to the chunk
        start_time = i * 2  # 2 seconds per chunk
        chunk_text = " ".join(words[i * chunk_size:(i + 1) * chunk_size])
        transcription.append((start_time, chunk_text))

    return transcription

# Convert timestamp to seconds
def timestamp_to_seconds(timestamp):
    minutes, seconds = map(int, timestamp.split(':'))
    return minutes * 60 + seconds

# Function to detect harmful words and return the timestamps
def find_harmful_word_timestamps(transcript_lines, timestamps):
    mute_times = []
    for i, line in enumerate(transcript_lines):
        words = line.split()
        for word in words:
            if word.lower() in harmful_words:
                mute_times.append(timestamps[i])
                break  # Only need to detect the first harmful word in each line
    return mute_times

# Function to mute sections of the video
def mute_video_at_timestamps(video_file, mute_times):
    video = VideoFileClip(video_file)
    clips = []
    last_end = 0

    # Process each mute interval
    for timestamp in mute_times:
        start_time = timestamp_to_seconds(timestamp)
        end_time = start_time + 1  # Mute for 1 second around the word (adjust as necessary)

        # Clip before the harmful word
        if start_time > last_end:
            clips.append(video.subclip(last_end, start_time))

        # Mute this segment
        muted_clip = video.subclip(start_time, end_time).volumex(0)
        clips.append(muted_clip)
        last_end = end_time

    # Append the remaining part of the video
    if last_end < video.duration:
        clips.append(video.subclip(last_end, video.duration))

    # Concatenate all the clips
    final_clip = concatenate_videoclips(clips)

    # Save the new muted video
    final_clip.write_videofile("output_muted_video.mp4", codec="libx264", audio_codec="aac")

def process_video(video_file):
    """
    Main function to process the video by extracting audio, transcribing it using Whisper, detecting harmful words,
    and muting the sections with harmful words in the final output video.
    """
    # Step 1: Extract audio from the video
    audio_path = extract_audio(video_file)
    print("Audio extracted successfully.")

    # Step 2: Transcribe the extracted audio using Whisper
    transcribed_text = transcribe_audio_whisper(audio_path)
    print("Audio transcribed using Whisper successfully.")

    # Step 3: Convert the transcription into timestamped text (2-second intervals)
    transcription = convert_audio_to_text_with_timestamps_whisper(audio_path, transcribed_text)

    # Clean up the audio file after use
    os.remove(audio_path)

    # Step 4: Prepare transcript data
    timestamps = [f"{int(ts // 60):02d}:{int(ts % 60):02d}" for ts, _ in transcription]
    transcript_lines = [text for _, text in transcription]

    # Step 5: Find harmful word timestamps
    mute_times = find_harmful_word_timestamps(transcript_lines, timestamps)

    # Step 6: Mute the video at the detected timestamps
    mute_video_at_timestamps(video_file, mute_times)
    print("Video processed and harmful words muted successfully.")

if __name__ == "__main__":
    video_file = "/content/videoplayback.mp4"
    process_video(video_file)