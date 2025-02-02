# Hackathon 2025

## Team Members
- **Khaled Saleh**  
- **Moetaz Mohamed**  
- **Mario El Shaer**  
- **Tarek Elalfi**  

---

## Project Setup

### 1. Install Dependencies
Run the following command in the **root directory** to install all required dependencies:

```bash
pip install -r requirements.txt
```

### 2. Set Up FFmpeg
FFmpeg is required to run voice commands in this project. Follow the steps below to set it up:

1. **Download FFmpeg:**  
   [Download FFmpeg](https://ffmpeg.org/download.html) and extract it to a folder (e.g., `C:\ffmpeg`).

2. **Add FFmpeg to Environment Variables:**
   - Open **System Properties** > **Environment Variables**.
   - Under **System variables**, find and select `Path`, then click **Edit**.
   - Click **New** and add the following path:
     ```
     C:\ffmpeg\bin
     ```
   - Click **OK** to save.

3. **Verify Installation:**
   Open a new terminal or command prompt and run:
   ```bash
   ffmpeg -version
   ```
   If FFmpeg is correctly installed, you'll see version information.

---
