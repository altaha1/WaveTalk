import tkinter as tk
from tkinter.ttk import *
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import pyttsx3
import customtkinter
import time

# Global Variables
cap = None # Holds the camera object
is_camera_on = False
frame_count = 0
model = YOLO('nano_vowels_1.pt')

last_detected_sign = None
sign_count = 0 # Counter for consecutive detections of the same sign
sign_threshold = 10 # Initial threshold of consecutive detections to translate a sign

font_scale = 0.8
thickness = 2
stroke_thickness = 6
font = cv2.FONT_HERSHEY_SIMPLEX

# Define variables for FPS calculation
start_time = time.time()
frame_count = 0
fps = 0


# read classes from text file
def read_classes(file_path):
    with open(file_path, 'r') as file:
        classes = [line.strip() for line in file]
    return classes

# Turn on the camera
def start_cam():
    global cap, is_camera_on
    if not is_camera_on:
        cap = cv2.VideoCapture(0)
        is_camera_on = True
        update_canvas()

# Turn off the camera
def stop_cam():
    global cap, is_camera_on, canvas
    if cap is not None:
        cap.release()
        is_camera_on = False
        canvas.img = initial_photo
        canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

# Clear text area
def clear_text():
    text_area.delete(1.0, tk.END)

# Speak text from the text area
def speak():
    engine = pyttsx3.init()
    engine.setProperty('rate', 170)
    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text_area.get(1.0, tk.END))
    engine.runAndWait()

# Display help window
def display_help():
    help_window = customtkinter.CTkToplevel(root)
    help_window.title("Help")
    
    # Load the help image
    help_image = Image.open("gestures_help.png")
    help_photo = ImageTk.PhotoImage(image=help_image)
    
    # Display the image
    help_label = customtkinter.CTkLabel(help_window, image=help_photo, text="")
    help_label.image = help_photo
    help_label.pack()

    help_window.attributes("-topmost", True)

# Quitting the application
def quit_app():
    stop_cam()
    root.quit()
    root.destroy()

def change_appearance_mode(new_appearance_mode: str):
    customtkinter.set_appearance_mode(new_appearance_mode)

# Toggle the camera state
def toggle_cam():
    global cam_state, cam_menu
    if cam_state == "Camera On":
        stop_cam()
        cam_state = "Camera Off"

    else:
        start_cam()
        cam_state = "Camera On"
    cam_menu.configure(text=cam_state)

# Change threshold for sign translation
def change_threshold(value):
    global sign_threshold
    # Calculate the inverse relationship between user's speed perception and sign_threshold
    value = int(value)
    sign_threshold = 31 - value
    threshold_label.configure(text="Translation Speed: " + str(value))

# Continiously update canvas with 
def update_canvas():
    global is_camera_on, frame_count, last_detected_sign, sign_count, fps, start_time, sign_threshold
    if is_camera_on:
        ret, frame = cap.read()
        if ret:
            frame_count += 1

            elapsed_time = time.time() - start_time
            
            # Calculate FPS
            if elapsed_time >= 1:
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0

            cv2.putText(frame, f"FPS: {int(fps)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            results = model.predict(frame, conf=0.70)
            a = results[0].boxes.data

            # Operations on detected class
            for x1, y1, x2, y2, conf, class_digit in a:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                c = class_list[int(class_digit)]

                text = f'{c} {conf: .2f}'

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, text, (x1, y1), font, font_scale, (0, 0, 0), stroke_thickness)
                cv2.putText(frame, text, (x1, y1),font, font_scale, (255, 255, 255), thickness)


                if c == last_detected_sign:
                    sign_count += 1
                else:
                    sign_count = 0
                
                # Check if sign has been detected for consecutive frames exceeding threshold
                if sign_count >= sign_threshold:
                    if c == "delete":
                        text_area.delete("end-2c")
                    elif c == "full_stop":
                        text_area.insert(tk.END, ".")
                    elif c == "space":
                        text_area.insert(tk.END, " ")
                    else:
                        text_area.insert(tk.END, str(c))
                    sign_count = 0

                last_detected_sign = c

            display_frame(frame)


        canvas.after(1, update_canvas)

# Display  frame on canvas
def display_frame(frame):

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    canvas_width = 640
    canvas_height = 360

    frame_resized = cv2.resize(frame_rgb, (canvas_width, canvas_height))

    photo = ImageTk.PhotoImage(image=Image.fromarray(frame_resized))
    
    canvas.img = photo
    canvas.config(width=canvas_width, height=canvas_height)
    canvas.create_image(canvas_width / 2, 0, anchor=tk.N, image=photo)

# Read classes from file
class_list = read_classes('sign_classes.txt')

# Set appearance mode and default color theme
customtkinter.set_appearance_mode("Dark")
customtkinter.set_default_color_theme("blue")

# Initialize root window
root = customtkinter.CTk()
root.title("WaveTalk")

# Frame for menu options
menu_frame = customtkinter.CTkFrame(root)
menu_frame.pack(fill='x')

# Option menu for appearance mode
appearance_mode_menu = customtkinter.CTkOptionMenu(menu_frame, values=["Dark", "Light"], command=change_appearance_mode)
appearance_mode_menu.pack(side='left', padx=0, pady=10)

# Button to show help
help_menu = customtkinter.CTkButton(menu_frame, text="Show Gestures", command=display_help)
help_menu.pack(side='left', padx=(10,0), pady=10)

# Button to toggle camera state
cam_state = "Camera Off"
cam_menu = customtkinter.CTkButton(menu_frame, text=cam_state, command=toggle_cam)
cam_menu.pack(side='left', padx=(10,0), pady=10)

# Canvas for displaying camera feed
canvas = customtkinter.CTkCanvas(root, width=640, height=360)
canvas.pack(fill='both', expand=True)

# Slider to adjust sign_threshold
threshold_slider = customtkinter.CTkSlider(root, from_=1, to=30, command=change_threshold)
threshold_slider.set(sign_threshold)
threshold_slider.pack(pady=(20, 5))

# Label to display current sign threshold
threshold_label = customtkinter.CTkLabel(root, text="Translation Speed: " + str(threshold_slider.get()), font=('Arial', 17, 'bold'))
threshold_label.pack(pady=(0, 5))

# Frame for text area
text_frame = customtkinter.CTkFrame(root)
text_frame.pack(fill='x')

# Text area for displaying recognized text
text_area = customtkinter.CTkTextbox(text_frame, height=130, corner_radius=20, wrap='word', font=('Arial', 17))
text_area.pack(fill='x', expand=True, padx=30, pady=(30,10))

# Frame for buttons
buttons_frame = customtkinter.CTkFrame(root)
buttons_frame.pack(fill='x')

# Calculate responsive padding values
padding_x = int(root.winfo_screenwidth() * 0.03)  # 3% of screen width
padding_y = int(root.winfo_screenheight() * 0.05)  # 5% of screen height

# Calculate responsive button width and height
button_width = int(root.winfo_screenwidth() * 0.1)  # 10% of screen width
button_height = int(root.winfo_screenheight() * 0.08)  # 8% of screen height

# Button to speak text from text area
speak_button = customtkinter.CTkButton(buttons_frame, text="Speak", command=speak, font=('Arial', 18),
                                       width=button_width, height=button_height)
speak_button.pack(side='left', padx=(padding_x, padding_x), pady=(padding_y, padding_y), expand=True)

# Button to clear text area
clear_button = customtkinter.CTkButton(buttons_frame, text="Clear", command=clear_text, font=('Arial', 18),
                                       width=button_width, height=button_height)
clear_button.pack(side='right', padx=(padding_x, padding_x), pady=(padding_y, padding_y), expand=True)

# Load initial image for canvas when camera is off
initial_image = Image.open('cam_not_connected.png')
initial_photo = ImageTk.PhotoImage(image=initial_image)
canvas.img = initial_photo
canvas.create_image(0, 0, anchor=tk.NW, image=initial_photo)

# Determine the screen width and height
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

# Calculate window position to center it on the screen
x = (screen_width - root.winfo_reqwidth()) // 2
y = (screen_height - root.winfo_reqheight()) // 3

# Set the geometry of the window to center it on the screen
root.geometry(f"+{x}+{y}")

root.resizable(False, False)
root.mainloop()
