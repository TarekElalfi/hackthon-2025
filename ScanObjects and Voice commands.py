import argparse
import queue
import sys
import sounddevice as sd
import time
import cv2
from vosk import Model, KaldiRecognizer
from ultralytics import YOLO
from elevenlabs.client import ElevenLabs
from elevenlabs import play
from dotenv import load_dotenv
import os
from collections import defaultdict
from threading import Thread
import easyocr  # Required for OCR functionality
import re


class VoiceObjectScanner:
    def __init__(self, model_lang="en-us", scan_duration=5):
        # Load ElevenLabs API key
        load_dotenv()
        self.client = ElevenLabs(api_key=("sk_a8247cec499ca17c0d1833516844a27edddb0287b8c54d56"))

        # Initialize VOSK speech recognition model
        self.model = Model(lang=model_lang)
        self.recognizer = None

        # Initialize YOLO model
        self.yolo_model = YOLO("yolov8n.pt")

        # Audio recording setup
        self.q = queue.Queue()
        self.listening = True
        self.last_command = None
        self.last_command_time = 0

        # Scanning parameters
        self.scan_duration = scan_duration
        self.IOU_THRESHOLD = 0.5  # Intersection over Union threshold
        self.object_history = {}
        self.summary_counts = defaultdict(int)

        # For position tracking
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(3))
        self.frame_height = int(self.cap.get(4))
        self.THIRD_WIDTH = 1 / 3
        self.THIRD_HEIGHT = 1 / 3
        self.scanning_active = False  # Controls when to scan

    def audio_callback(self, indata, frames, time, status):
        """Handles incoming audio."""
        if status:
            print(status, file=sys.stderr)
        self.q.put(bytes(indata))

    def transcribe_and_store(self):
        """Continuously listens for voice commands."""
        self.synthesize_speech("Welcome to see through, You can say the following commands to help you. Say read to capture a text infront of you and read it for you. Say scan to scan your surrounding objects, say safety to detecte proximity objects detection")
        while self.listening:
            data = self.q.get()
            if self.recognizer.AcceptWaveform(data):
                result = self.recognizer.Result()
                text = result.split('"text" : "')[1].split('"')[0] if '"text" : "' in result else ""

                if text:
                    print(f"Recognized: {text}")

                    # Normalize text to lowercase for consistent matching
                    text_lower = text.lower()

                    if re.search(r'\b(scan|scam|scanner|scanned|scanners|scandal)\b', text_lower):
                        self.trigger_scan()

                    #if text.lower() == "read" You are to add this
                    if re.search(r'\b(read|reader|reading|red|either)\b', text_lower):
                        self.trigger_read()
                        # Add trigger calls in the transcribe_and_store method
                    if self.listening and re.search(r'\b(safety|say tea|safely)\b', text_lower):
                        # self.listening = False 
                        self.trigger_proximity()

                    if text.lower() == "stop":
                        self.trigger_stop()

    def trigger_scan(self):
        if self.last_command != "scan" or time.time() - self.last_command_time > 3:
            print("Starting scan...")
            self.synthesize_speech("Starting the scan.")
            self.last_command = "scan"
            self.last_command_time = time.time()
            self.scanning_active = True  # Enable scanning

    def trigger_read(self):
        """Captures an image from the camera and reads any detected text aloud."""
        print("Capturing image for text reading...")
        self.synthesize_speech("Capturing image for text reading.")

        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture image.")
            self.synthesize_speech("Failed to capture image.")
            return

        # Initialize EasyOCR reader
        reader = easyocr.Reader(['en'])

        # Perform OCR on the captured frame
        results = reader.readtext(frame)

        if results:
            text_content = " ".join([result[1] for result in results])
            print(f"Detected Text: {text_content}")
            self.synthesize_speech(f"The text says: {text_content}")
        else:
            print("No readable text detected.")
            self.synthesize_speech("No readable text detected.")


    def trigger_stop(self):
        print("Stopping the program.")
        self.synthesize_speech("Stopping the program.")
        self.listening = False

    def synthesize_speech(self, text):
        """Converts text to speech using ElevenLabs."""
        audio = self.client.text_to_speech.convert(
            text=text,
            voice_id="JBFqnCBsd6RMkjVDRZzb",
            model_id="eleven_multilingual_v2",
            output_format="mp3_44100_128",
        )
        play(audio)

    def iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1, y1, x2, y2 = box1
        x1_p, y1_p, x2_p, y2_p = box2

        # Intersection
        inter_x1 = max(x1, x1_p)
        inter_y1 = max(y1, y1_p)
        inter_x2 = min(x2, x2_p)
        inter_y2 = min(y2, y2_p)
        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

        # Union
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x2_p - x1_p) * (y2_p - y1_p)
        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def run_camera_feed(self):
        """Continuously displays the camera feed and handles scanning."""
        while self.listening:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame.")
                break

            if self.scanning_active:
                self.perform_scan(frame)
            else:
                cv2.imshow("YOLO Camera Feed", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.trigger_stop()
                break
            elif key == ord('r'):
                self.trigger_read()            
            elif key == ord('s'):
                self.trigger_scan()
                    # Add keyboard shortcut in run_camera_feed
            elif self.listening and key == ord('p'):
                # self.listening = False 

                self.trigger_proximity()


        self.cap.release()
        cv2.destroyAllWindows()

    def perform_scan(self, frame):
        """Runs YOLO object detection and summarizes results."""
        start_time = time.time()
        self.summary_counts.clear()
        self.object_history.clear()

        while time.time() - start_time < self.scan_duration and self.listening:
            ret, frame = self.cap.read()
            if not ret:
                break

            results = self.yolo_model.predict(frame, conf=0.5, verbose=False)
            current_frame_objects = defaultdict(int)
            current_time = time.time()

            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, class_id = box.tolist()
                    object_name = self.yolo_model.names[int(class_id)]

                    # Determine position
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    position = self.get_position(center_x, center_y)

                    object_box = (x1, y1, x2, y2)
                    key = f"{object_name} in {position}"

                    is_new_object = True
                    for prev_box, prev_time in self.object_history.get(object_name, []):
                        if self.iou(object_box, prev_box) > self.IOU_THRESHOLD and (current_time - prev_time < self.scan_duration):
                            is_new_object = False
                            break

                    if is_new_object:
                        current_frame_objects[key] += 1
                        if object_name not in self.object_history:
                            self.object_history[object_name] = []
                        self.object_history[object_name].append((object_box, current_time))

                    # Draw bounding boxes
                    label = f"{object_name} ({int(conf * 100)}%)"
                    color = (0, 255, 0)
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                    cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if current_frame_objects:
                for key, count in current_frame_objects.items():
                    self.summary_counts[key] += count

            cv2.imshow("YOLO Camera Feed", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.trigger_stop()
                break

        self.summarize_results()
        self.scanning_active = False  # Disable scanning after completion
    def trigger_proximity(self):
        """Checks for objects that are too close and announces warnings."""
        print("Activating proximity mode...")
        self.synthesize_speech("Activating proximity mode for safety.")

        start_time = time.time()
        FRAME_AREA = self.frame_width * self.frame_height
        PROXIMITY_THRESHOLD = 0.66 * FRAME_AREA  # 66% of the frame area

        self.proximity_active = True  # Flag to control proximity mode
        self.listening = True        # Disable other commands during proximity mode

        while time.time() - start_time < 10:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to capture frame. Reinitializing camera...")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)  # Reinitialize the camera
                continue

            # Object detection using YOLO
            results = self.yolo_model.predict(frame, conf=0.5, verbose=False)
            objects_too_close = False

            for result in results:
                for box in result.boxes.data:
                    x1, y1, x2, y2, conf, class_id = box.tolist()
                    object_area = (x2 - x1) * (y2 - y1)

                    if object_area >= PROXIMITY_THRESHOLD:
                        objects_too_close = True
                        break  # No need to check further if one object is too close

                if objects_too_close:
                    break

            # Continuous feedback based on proximity
            if objects_too_close:
                print("⚠️ Watch out! Objects ahead.")
                self.synthesize_speech("Watch out! Objects ahead.")
            else:
                print("✅ Safe ahead.")
                self.synthesize_speech("Safe ahead.")

            cv2.imshow("YOLO Camera Feed", frame)  # Reuse the main feed window

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                self.trigger_stop()
                break

        print("Proximity mode deactivated.")
        self.synthesize_speech("Proximity mode deactivated.")
        self.proximity_active = False  # Reset flag after exiting proximity mode
        self.listening = True         # Re-enable listening for commands

    def summarize_results(self):
        """Reads out the summary of detected objects."""
        if self.summary_counts:
            summary = ", ".join([f"{count} {name}" for name, count in self.summary_counts.items()])
            print(f"🟢 Summary (last {self.scan_duration} sec): {summary}")
            self.synthesize_speech(f"Summary of the last {self.scan_duration} seconds: {summary}")
        else:
            print("🟢 No objects detected during the scan.")
            self.synthesize_speech("No objects detected during the scan.")

    def get_position(self, center_x, center_y):
        """Determine the object's horizontal position in the frame (left, middle, right)."""
        if center_x < self.frame_width * self.THIRD_WIDTH:
            return "left"
        elif center_x > self.frame_width * (2 * self.THIRD_WIDTH):
            return "right"
        else:
            return "middle"


    def start_listening(self, device=None, samplerate=None):
        """Starts voice recognition and camera feed."""
        try:
            if samplerate is None:
                device_info = sd.query_devices(device, "input")
                samplerate = int(device_info["default_samplerate"])

            self.recognizer = KaldiRecognizer(self.model, samplerate)

            # Start camera feed in a separate thread
            Thread(target=self.run_camera_feed).start()

            # Start voice recognition
            with sd.RawInputStream(samplerate=samplerate, blocksize=8000, device=device,
                                   dtype="int16", channels=1, callback=self.audio_callback):
                print("Listening for commands... Say 'scan' or 'stop' or press 's' to scan and 'q' to stop.")
                self.transcribe_and_store()

        except KeyboardInterrupt:
            print("\nStopped manually.")
        except Exception as e:
            print(f"Error: {e}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--device", type=int, help="Input device ID")
    parser.add_argument("-r", "--samplerate", type=int, help="Sampling rate")
    parser.add_argument("-m", "--model", type=str, default="en-us", help="Language model")
    args = parser.parse_args()

    scanner = VoiceObjectScanner(model_lang=args.model)
    scanner.start_listening(device=args.device, samplerate=args.samplerate)
