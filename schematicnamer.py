import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import shutil

OUTPUT_DIRECTORY = 'data/schematics/trees'


class Watcher:
    # Replace with your Minecraft schematics directory path
    DIRECTORY_TO_WATCH = r"C:\Users\mmmfr\curseforge\minecraft\Instances\Litematica\config\worldedit\schematics"

    def __init__(self):
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(
            event_handler, self.DIRECTORY_TO_WATCH, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            return None

        elif event.event_type == 'modified':
            # When a file is created, copy it with a timestamp in its name.
            print(f"Received created event - {event.src_path}.")
            # Creating a unique timestamp
            timestamp = time.strftime("%Y%m%d%H%M%S")
            # New file name with timestamp
            new_file_name = os.path.join(OUTPUT_DIRECTORY,
                                         os.path.splitext(os.path.basename(event.src_path))[0] +
                                         "_" + timestamp +
                                         os.path.splitext(event.src_path)[1])
            shutil.copy(event.src_path, new_file_name)  # Copy the file
            print(f"Copied file to {new_file_name}")


if __name__ == '__main__':
    w = Watcher()
    w.run()
