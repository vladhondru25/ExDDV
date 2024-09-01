import configparser
import json
import os
import threading
import time
import signal
from PIL import Image, ImageTk

import cv2
import numpy as np
import tkinter as tk
from tkinter import Label, Button, messagebox

from database import DatabaseConnector
from utils import on_ctrl_c_signal, cleanup_and_exit


DATASET = "Farceforensics++"


class VideoPlayerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Video Player")

        self.cfg = configparser.ConfigParser()
        self.cfg.read('config.cfg')

        self.movies_path = self.cfg["USER_DATA"]["MOVIES_PATH"]
        self.original_movies_path = self.cfg["USER_DATA"]["ORIGINAL_MOVIES_PATH"]
        self.username = self.cfg["USER_DATA"]["USERNAME"]

        # Dabase connector
        self.database_conn = DatabaseConnector(self.cfg)

        self.width = 500
        self.height = 400
        
        # List of video pairs (source, target)
        video_names = os.listdir(self.movies_path)
        # This is used to select Eduard's or Vlad's videos
        if self.username == "Eduard Hogea":
            video_names = [vid_name for vid_name in video_names if int(vid_name.split("_")[0]) < 500]
        else:
            video_names = [vid_name for vid_name in video_names if int(vid_name.split("_")[0]) >= 500]

        print(f"Original videos: {len(video_names)}")

        videos_annotated = self.database_conn.read_movie_entries(self.username)
        videos_not_annotated = set(video_names).difference(videos_annotated) 
        video_names = list(videos_not_annotated)
        print(f"Remaining videos: {len(video_names)}")

        video_pairs_1 = [os.path.join(self.movies_path, video_name) for video_name in video_names]
        video_pairs_2 = [os.path.join(self.original_movies_path, f"{video_name.split('_')[0]}.mp4") for video_name in video_names]

        self.video_pairs = list(zip(video_pairs_1, video_pairs_2))

        self.current_index = 0
        self.show_video_2 = False  # Initially hide the second video

        # Create labels to hold the videos
        self.video_label_1 = Label(self.root)
        self.video_label_1.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.video_label_2 = Label(self.root)
        self.video_label_2.grid(row=0, column=3, columnspan=3, padx=10, pady=10)

        # Create the functionality of storing click locations
        self.video_label_1.bind("<Button-1>", self.on_click)
        self.click_locations = {}

        # Control buttons
        self.prev_button = Button(self.root, text="Previous", command=self.show_previous)
        self.prev_button.grid(row=1, column=0, padx=10, pady=10)

        self.restart_button = Button(self.root, text="Restart video", command=self.restart_video)
        self.restart_button.grid(row=1, column=1, padx=10, pady=10)

        self.next_button = Button(self.root, text="Next", command=self.show_next)
        self.next_button.grid(row=1, column=2, padx=10, pady=10)

        self.reveal_button = Button(self.root, text="Reveal/Hide Target Video", command=self.reveal_hide_video)
        self.reveal_button.grid(row=1, column=4, padx=10, pady=10)

        # Add Text Box and Submit Button
        self.input_label = Label(self.root, text="Enter Text:")
        self.input_label.grid(row=2, column=0, pady=5)

        # self.text_box = ttk.Entry(self.root, width=50)
        self.text_box = tk.Text(self.root, height=7, width=40, font=('Arial', 14))
        self.text_box.grid(row=2, column=1, pady=5)

        self.submit_button = Button(self.root, text="Submit", command=self.submit_text)
        self.submit_button.grid(row=2, column=2, pady=5)

        # Flags to control video threads
        self.stop_threads = False
        self.thread = None

        # Load the initial video pair
        self.load_videos()

    def load_videos(self):
        # Get the current video paths
        source_video, target_video = self.video_pairs[self.current_index]

        # Stop any currently running videos
        self.stop_current_videos()

        # Start the thread to play the new videos
        self.stop_threads = False
        self.thread = threading.Thread(target=self.play_videos, args=(source_video, target_video))
        self.thread.start()

    def stop_current_videos(self):
        # Signal the threads to stop
        self.stop_threads = True

        # Join the threads to ensure they have stopped before loading new videos
        if self.thread is not None:
            self.thread.join()

        self.thread = None

    def play_videos(self, video_path_1, video_path_2):
        cap1 = cv2.VideoCapture(video_path_1)
        cap2 = cv2.VideoCapture(video_path_2)
        
        # Determine the frame rate to sync the playback
        fps1 = cap1.get(cv2.CAP_PROP_FPS) or 24  # Default to 30 FPS if FPS is not available
        fps2 = cap2.get(cv2.CAP_PROP_FPS) or 24
        sync_fps = min(fps1, fps2)  # Sync both videos to the slower frame rate
        delay = int(1000 / sync_fps)  # Delay in milliseconds between frames

        self.frame_idx = 0
        while not self.stop_threads:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()

            if not ret1 or not ret2 or self.stop_threads:
                break  # End of video or stop signal received

            # Resize frames to fit the labels
            frame1 = cv2.resize(frame1, (self.width, self.height))
            if self.show_video_2:
                frame2 = cv2.resize(frame2, (self.width, self.height))
            else:
                frame2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)

            # Convert frames to RGB format (Tkinter uses RGB, OpenCV uses BGR)
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            # Convert the frames to PIL Images, then to Tkinter PhotoImages
            img1 = Image.fromarray(frame1)
            img2 = Image.fromarray(frame2)

            imgtk1 = ImageTk.PhotoImage(image=img1)
            imgtk2 = ImageTk.PhotoImage(image=img2)

            # Update the labels with the new frames
            self.video_label_1.imgtk = imgtk1  # Prevent garbage collection
            self.video_label_1.config(image=imgtk1)

            self.video_label_2.imgtk = imgtk2  # Prevent garbage collection
            self.video_label_2.config(image=imgtk2)

            # Wait for the next frame, based on the synchronized frame rate
            time.sleep(delay / 1000.0)

            self.frame_idx += 1

        cap1.release()
        cap2.release()

    def restart_video(self):
        self.load_videos()

    def show_next(self):
        if self.current_index < len(self.video_pairs) - 1:
            self.click_locations = {}

            self.current_index += 1
            self.show_video_2 = False  # Hide the second video initially for the previous pair
            self.load_videos()

    def show_previous(self):
        if self.current_index > 0:
            self.click_locations = {}

            self.current_index -= 1
            self.show_video_2 = False  # Hide the second video initially for the previous pair
            self.load_videos()

    def reveal_hide_video(self):
        # Set flag to show the second video and reload the video frames
        self.show_video_2 = not self.show_video_2

    def submit_text(self):
        # Retrieve the text from the text box
        text = self.text_box.get("1.0", tk.END).strip()

        # In a real application, you would send the text to a server, database, etc.
        # For this example, we'll just print it to the console
        # print(f"Text submitted: {text}")
        movie_path = self.video_pairs[self.current_index][0]
        movie_path_head, movie_name_db = os.path.split(movie_path)
        _, manipulation_folder = os.path.split(movie_path_head)

        self.database_conn.add_row(
            user=self.username,
            video_name=movie_name_db,
            text=text,
            dataset=DATASET,
            manipulation=manipulation_folder,
            click_locations = json.dumps(self.click_locations)
        )

        # Clear the text box
        self.text_box.delete("1.0", tk.END)

        self.click_locations = {}

        # Show success message
        messagebox.showinfo("Success", "Text submitted successfully!")

    def on_click(self, event):
        # Get the coordinates of the click relative to the Label
        x, y = event.x, event.y
        self.click_locations[self.frame_idx] = {"x": x / self.width, "y": y / self.height}

    def close_db_connection(self):
        self.database_conn.close()


# Main application
if __name__ == "__main__":
    # Create the main Tkinter window
    root = tk.Tk()

    # Create the video player application
    app = VideoPlayerApp(root)

    # Bind the window close (X button) to the custom handler
    def _callback_destroy():
        cleanup_and_exit(root, app)
    root.protocol("WM_DELETE_WINDOW", _callback_destroy)

    # Setup signal handler for SIGINT (CTRL+C)
    def _callback_sigint(signal_received, frame):
        on_ctrl_c_signal(signal_received=signal_received, frame=frame, root=root, app=app)
    signal.signal(signal.SIGINT, _callback_sigint)
    try:
        root.mainloop()
    except KeyboardInterrupt:
        # Handle the KeyboardInterrupt in case it is not caught by signal
        on_ctrl_c_signal(signal_received=signal.SIGINT, frame=None, root=root, app=app)
