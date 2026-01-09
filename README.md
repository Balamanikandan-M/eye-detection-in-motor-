It is not a robot project by default — it is an AI eye-gesture detection system that sends HTTP commands.

It depends on MediaPipe Tasks FaceLandmarker, not Haar cascades.

It only sends two commands:

/forward when eyes are open

/stop when eyes closed > 0.5 s

The ESP32 part is external, not included in this repository.

So your correct project category is:

Real-Time Eye Gesture Detection with HTTP Control Interface
