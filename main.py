import configparser
import os
import signal
from PIL import Image, ImageTk

import imageio
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox

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
        self.username = self.cfg["USER_DATA"]["USERNAME"]

        # Dabase connector
        self.database_conn = DatabaseConnector(self.cfg)
        
        # Video list (replace with your own video file paths)
        self.video_list = [video_name for video_name in os.listdir(self.movies_path)]
        self.current_video_index = 0
        self.is_playing = False  # Flag to track if the video is playing

        videos_annotated = self.database_conn.read_movie_entries(self.username)
        videos_not_annotated = set(self.video_list).difference(videos_annotated) 
        print(f"Original videos: {len(self.video_list)}")
        self.video_list = list(videos_not_annotated)
        self.video_list = [os.path.join(self.movies_path, video_name) for video_name in self.video_list]
        print(f"Remaining videos: {len(self.video_list)}")

        # FPS (frames per second) control
        self.fps = 24  # You can adjust this to control the playback speed

        # Label to display the current video
        self.video_label = ttk.Label(self.root, text="Video Player", font=("Arial", 16))
        self.video_label.pack(pady=10)

        # Canvas to show video frames
        self.canvas = tk.Canvas(self.root, width=640, height=360)
        self.canvas.pack()

        # Control buttons
        self.previous_button = ttk.Button(self.root, text="Previous", command=self.show_previous_video)
        self.previous_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.restart_button = ttk.Button(self.root, text="Restart video", command=self.restart_video)
        self.restart_button.pack(side=tk.LEFT, padx=10, pady=10)

        self.next_button = ttk.Button(self.root, text="Next", command=self.show_next_video)
        self.next_button.pack(side=tk.RIGHT, padx=10, pady=10)

        # Add Text Box and Submit Button
        self.input_label = ttk.Label(self.root, text="Enter Text:")
        self.input_label.pack(pady=5)

        # self.text_box = ttk.Entry(self.root, width=50)
        self.text_box = tk.Text(self.root, height=7, width=40, font=('Arial', 14))
        self.text_box.pack(pady=5)

        self.submit_button = ttk.Button(self.root, text="Submit", command=self.submit_text)
        self.submit_button.pack(pady=5)

        # Load and play the initial video
        self.load_video(self.video_list[self.current_video_index])

    def load_video(self, video_path):
        if not os.path.exists(video_path):
            messagebox.showerror("Error", f"Video file not found: {video_path}")
            return
        
        self.video_label.config(text=f"Playing: {os.path.basename(video_path)}")
        self.reader = imageio.get_reader(video_path)

        self.is_playing = True
        self.current_frame = 0  # Start from the first frame
        self.play_video()

    def play_video(self):
        if self.is_playing:
            try:
                # Get the current frame
                frame = self.reader.get_data(self.current_frame)

                # Convert frame to Image and resize to fit canvas
                frame_image = Image.fromarray(frame)
                frame_image = frame_image.resize((640, 360))
                frame_photo = ImageTk.PhotoImage(frame_image)

                # Display frame on canvas
                self.canvas.create_image(0, 0, anchor=tk.NW, image=frame_photo)
                self.canvas.image = frame_photo

                # Update current frame
                self.current_frame += 1

                # Control FPS by adding delay based on desired FPS
                delay = int(1000 / self.fps)  # Convert FPS to milliseconds
                self.root.after(delay, self.play_video)

            except IndexError:
                # Reached end of video
                self.is_playing = False

    def restart_video(self):
        self.current_frame = 0
        self.is_playing = True
        self.play_video()

    def show_next_video(self):
        self.current_video_index = (self.current_video_index + 1) % len(self.video_list)
        self.load_video(self.video_list[self.current_video_index])

    def show_previous_video(self):
        self.current_video_index = (self.current_video_index - 1) % len(self.video_list)
        self.load_video(self.video_list[self.current_video_index])

    def submit_text(self):
        # Retrieve the text from the text box
        text = self.text_box.get("1.0", tk.END).strip()

        # In a real application, you would send the text to a server, database, etc.
        # For this example, we'll just print it to the console
        # print(f"Text submitted: {text}")
        movie_path = self.video_list[self.current_video_index]
        movie_path_head, movie_name_db = os.path.split(movie_path)
        _, manipulation_folder = os.path.split(movie_path_head)

        self.database_conn.add_row(
            user=self.username,
            video_name=movie_name_db,
            text=text,
            dataset=DATASET,
            manipulation=manipulation_folder
        )

        # # Clear the text box
        self.text_box.delete("1.0", tk.END)

        # Show success message
        messagebox.showinfo("Success", "Text submitted successfully!")

    def close_db_connection(self):
        self.database_conn.close()


# Main application
if __name__ == "__main__":
    root = tk.Tk()

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
