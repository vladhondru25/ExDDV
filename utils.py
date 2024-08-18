import sys


def on_ctrl_c_signal(signal_received, frame, root, app):
    # Function that runs when CTRL+C is pressed
    print("CTRL+C detected! Cleaning up...")

    cleanup_and_exit(root, app)

def cleanup_and_exit(root, app):
    # Close db connection
    app.close_db_connection()
    
    # Gracefully close the tkinter application
    root.quit()  # Stops the mainloop
    root.destroy()  # Destroys the tkinter window
    sys.exit(0)  # Exit the script with a success code