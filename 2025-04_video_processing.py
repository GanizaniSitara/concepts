#!/usr/bin/env python3
import os
from moviepy.editor import (
    VideoFileClip,
    ImageClip,
    TextClip,
    CompositeVideoClip,
    concatenate_videoclips,
    vfx
)

def get_video_files(directory, video_exts=(".mp4", ".mov", ".avi", ".mkv")):
    """Return a list of video files in the directory with specified extensions."""
    return [
        f for f in os.listdir(directory)
        if f.lower().endswith(video_exts) and os.path.isfile(os.path.join(directory, f))
    ]

def is_processed(filename, processed_suffix="_processed"):
    """Return True if the filename indicates a processed file."""
    root, _ = os.path.splitext(filename)
    return root.endswith(processed_suffix)

def get_processed_filename(filename, processed_suffix="_processed"):
    """Return the processed file name by inserting the processed_suffix before the extension."""
    root, ext = os.path.splitext(filename)
    return f"{root}{processed_suffix}{ext}"

def process_video(file_path, output_path, title, description, freeze_time=3.0):
    clip = VideoFileClip(file_path)
    first_frame = clip.get_frame(0)

    freeze_clip = ImageClip(first_frame).set_duration(freeze_time)
    freeze_clip = freeze_clip.fx(vfx.colorx, 0.7)

    # Title text (changed align="left" --> align="West")
    title_clip = TextClip(
        txt=title,
        fontsize=80,
        color="yellow",
        font="Arial-Bold",
        method="caption",
        align="West",  # <-- Key fix
        size=(int(clip.w * 0.7), None)
    ).set_duration(freeze_time).set_position((50, 50))

    # Description text (also align="West")
    desc_clip = TextClip(
        txt=description,
        fontsize=70,
        color="yellow",
        font="Arial-Bold",
        method="caption",
        align="West",
        size=(int(clip.w * 0.7), None)
    ).set_duration(freeze_time).set_position((50, clip.h - 150))

    composite_freeze = CompositeVideoClip([freeze_clip, title_clip, desc_clip], size=(clip.w, clip.h))
    final_clip = concatenate_videoclips([composite_freeze, clip])

    final_clip.write_videofile(
        output_path,
        codec="libx264",
        audio_codec="aac",
        fps=30,
        threads=4
    )

def main():
    base_dir = r"C:\vids"  # Change as needed
    processed_suffix = "_processed"

    while True:
        all_files = get_video_files(base_dir)
        processed_files, unprocessed_files = [], []

        for f in all_files:
            if is_processed(f, processed_suffix):
                processed_files.append(f)
            else:
                proc_name = get_processed_filename(f, processed_suffix)
                if proc_name in all_files:
                    processed_files.append(f)
                else:
                    unprocessed_files.append(f)

        print("\nProcessed files:")
        if processed_files:
            for idx, f in enumerate(processed_files, start=1):
                print(f"  {idx}. {f} -> {get_processed_filename(f, processed_suffix)}")
        else:
            print("  None")

        print("\nUnprocessed files:")
        if not unprocessed_files:
            print("  None. All files are processed!")
            break
        else:
            for idx, f in enumerate(unprocessed_files, start=1):
                print(f"  {idx}. {f}")

        user_input = input("\nEnter the number of the file you want to process (or 'exit' to quit): ").strip()
        if user_input.lower() in ("exit", "quit"):
            print("Exiting.")
            break

        try:
            sel_num = int(user_input)
            if sel_num < 1 or sel_num > len(unprocessed_files):
                print("Invalid selection. Try again.")
                continue
        except ValueError:
            print("Invalid input. Please enter a valid number or 'exit'.")
            continue

        selected_file = unprocessed_files[sel_num - 1]
        input_path = os.path.join(base_dir, selected_file)
        output_file = get_processed_filename(selected_file, processed_suffix)
        output_path = os.path.join(base_dir, output_file)

        # Prompt for metadata
        title = input("Enter the title for this video: ").strip()
        description = input("Enter the description for this video: ").strip()

        print(f"\nProcessing '{selected_file}' -> '{output_file}' with title '{title}' and description '{description}'.\n")
        process_video(input_path, output_path, title, description)
        print("\nProcessing complete!\n")

if __name__ == "__main__":
    main()
