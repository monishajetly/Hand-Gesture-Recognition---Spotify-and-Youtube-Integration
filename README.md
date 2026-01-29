# Hand-Gesture-Recognition---Spotify-and-Youtube-Integration
A real-time hand gesture recognition system built using Python, MediaPipe, and OpenCV. This project detects hand landmarks from a live camera feed and recognizes predefined gestures that can be extended to control Spotify and Youtube applications.

## Tech Stack

* **Python**
* **MediaPipe**
* **OpenCV**
* **NumPy**

## 📂 Project Structure

```
hand-gesture-recognition/
│
├── app.py                                      # Main script to run 
├── keypoint_classification_EN.py               # Gesture definitions
├── youtube_open.py                             # Youtube Application
├── youtube_play_pause.py
├── youtube_prev_video.py
├── youtube_skip_video.py
├── open_n_play.py                              # Spotify Application
├── play_n_pause.py
├── like_songs.py
├── skip_songs.py
├── prev_songs.py
├── vol_up.py
├── vol_down.py
├── vol_down.py                                 # Generating Playlists in Spotify     
├── classical_songs.py              
├── happy_songs.py
├── gen_songs.py
├── pop_songs.py
├── rock_songs.py
└── sad_songs.py                                           
```

## How It Works

1. Captures video input from webcam
2. Detects hands using MediaPipe
3. Extracts hand landmarks (21 points)
4. Analyzes landmark positions
5. Classifies gestures
6. User decides which application to control (Spotify/Youtube)
7. Youtube Application - Can control basic functions (like play, pause and skip video)
8. Spotify Application - Can control basic functions (play, pause, prev, like, skip, increase and decrease volume) and Generate playlist based on genre (classical, rock, happy, sad and pop)

## Future Enhancements

* Creating a User Interface
  
## 👩‍💻 Author

**Monisha**

